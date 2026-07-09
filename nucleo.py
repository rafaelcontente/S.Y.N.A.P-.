"""Núcleo Cognitivo de Dupla Via com Deteção de Artefactos (Loop Neural-Simbólico + Autoencoder).

Módulo 5 do S.Y.N.A.P. Compõe as peças públicas já validadas dos
Módulos 3 (`expansor`) e 4 (`hipotalamo`) num único ciclo de geração
por linha: cada linha quimera candidata passa sucessivamente pelo
filtro de Mahalanobis, pelo score de plausibilidade, pela validação
lógica estrita (ASP) e, finalmente, por um autoencoder que deteta
"estranheza" via perda de reconstrução. Qualquer rejeição — lógica ou
de artefacto — penaliza de imediato as fontes envolvidas, fazendo com
que as primeiras rejeições de um lote modelem ativamente as remisturas
seguintes, dentro do mesmo lote.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from synap.expansor.expansor import (
    DEFAULT_INTERNAL_PENALTY,
    DEFAULT_MAHALANOBIS_THRESHOLD,
    DEFAULT_PLAUSIBILITY_THRESHOLD,
    GLOBAL_BUDGET_MULTIPLIER,
    K_MAX,
    K_MIN,
    MAX_ATTEMPTS_PER_ROW,
    MAX_MEMORY_SIZE,
    build_target_distribution,
    dag_column_groups,
    generate_chimera_row,
    mahalanobis_distance,
    penalize_sources,
    train_plausibility_model,
)
from synap.hipocampo.models import CausalDAG
from synap.hipotalamo.hipotalamo import validate_row
from synap.neocortex.models import ContinuousStatistics, StatisticalContract

from .exceptions import InsufficientDataError, NucleoValidationError
from .models import AntiExample, AutoencoderBaseline, NucleoBatchReport, NucleoOutput, NucleoRejectionReason

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

MIN_SEED_SIZE_AUTOENCODER: int = 50
"""Tamanho mínimo da semente para treinar o autoencoder de forma estável."""

DEFAULT_STRANGENESS_STD: float = 3.0
"""Nº de desvios padrão acima da média de perda que classifica uma linha como ESTRANHA."""

MIN_LOSS_STD_FLOOR: float = 1e-6
"""Piso do desvio padrão da perda, evita um limiar degenerado (=média)."""

DEFAULT_ASP_PENALTY: float = 0.6
"""Fator de penalização aplicado às fontes de uma rejeição lógica (ASP).

Menos agressivo do que a penalização de artefacto: uma violação lógica
pode ser um evento isolado do valor amostrado, não necessariamente uma
característica sistemática das fontes envolvidas."""

DEFAULT_ARTIFACT_PENALTY: float = 0.3
"""Fator de penalização aplicado às fontes de um artefacto (autoencoder).

Mais agressivo do que a penalização ASP: um artefacto reflete uma
combinação que o modelo de "realismo" (autoencoder) nunca viu na
semente — maior confiança de que é a combinação de fontes, não o
acaso, que está a produzir estranheza."""

DEFAULT_AUTOENCODER_UPDATE_INTERVAL: int = 500
"""Nº de linhas aprovadas entre atualizações incrementais (`partial_fit`) do autoencoder."""

AUTOENCODER_MAX_ITER: int = 2_000
AUTOENCODER_TRAINING_RESTARTS: int = 3
"""Nº de reinícios de treino com inicializações distintas; fica-se com o
modelo de menor perda de reconstrução média — mitiga mínimos locais
fracos do MLP, especialmente com gargalos muito estreitos (dim=1)."""


# --------------------------------------------------------------------------
# Autoencoder (MLP com gargalo, treinado a reconstruir o próprio input)
# --------------------------------------------------------------------------


class AutoencoderScorer:
    """Deteta "estranheza" via perda de reconstrução de um autoencoder leve.

    O autoencoder (MLPRegressor com uma camada de gargalo, treinado a
    reconstruir o seu próprio input) aprende a variedade de baixa
    dimensão ocupada pelas combinações reais da semente. Combinações
    fora dessa variedade — mesmo que cada valor seja marginalmente
    plausível — são mal reconstruídas, produzindo perda elevada.
    """

    def __init__(
        self,
        model: MLPRegressor,
        cat_encoders: dict[str, dict[Any, int]],
        numeric_cols: list[str],
        categorical_cols: list[str],
        num_means: dict[str, float],
        num_stds: dict[str, float],
        baseline: AutoencoderBaseline,
    ) -> None:
        self.model = model
        self._cat_encoders = cat_encoders
        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols
        self._num_means = num_means
        self._num_stds = num_stds
        self.baseline = baseline

    def _encode(self, rows: list[dict[str, Any]]) -> np.ndarray:
        partes = []
        if self._numeric_cols:
            partes.append(
                np.array(
                    [
                        [
                            (float(r[c]) - self._num_means[c]) / self._num_stds[c]
                            for c in self._numeric_cols
                        ]
                        for r in rows
                    ]
                )
            )
        for coluna in self._categorical_cols:
            encoder = self._cat_encoders[coluna]
            codigos = np.array([[encoder.get(r[coluna], -1)] for r in rows], dtype=float)
            partes.append(codigos)
        return np.hstack(partes) if partes else np.zeros((len(rows), 0))

    def reconstruction_loss(self, row: dict[str, Any]) -> float:
        """Calcula a perda de reconstrução (MSE) de `row`."""
        x = self._encode([row])
        x_reconstruido = self.model.predict(x)
        return float(np.mean((x - x_reconstruido) ** 2))

    def is_strange(self, row: dict[str, Any]) -> bool:
        """Classifica `row` como ESTRANHA se a sua perda exceder o limiar da linha de base."""
        return self.reconstruction_loss(row) > self.baseline.limiar

    def partial_update(self, rows: pd.DataFrame) -> None:
        """Atualiza incrementalmente os pesos do autoencoder com novas linhas aprovadas.

        A linha de base (média/desvio/limiar) NÃO é recalculada aqui —
        mantém-se fixa na semente original, por decisão deliberada: um
        limiar em deriva contínua enfraqueceria a deteção de artefactos
        ao longo de um lote longo (o próprio autoencoder aprenderia a
        "normalizar" os artefactos que devia detetar).
        """
        x = self._encode([row for _, row in rows.iterrows()])
        if x.shape[0] == 0:
            return
        self.model.partial_fit(x, x)


def train_autoencoder(
    seed: pd.DataFrame,
    contract: StatisticalContract,
    strangeness_std: float = DEFAULT_STRANGENESS_STD,
    rng_seed: int | None = None,
) -> AutoencoderScorer:
    """Treina o autoencoder (Módulo 5) na semente e fixa a sua linha de base.

    Args:
        seed: A semente do Módulo 2 (ou memória acumulada), usada como
            único exemplo de "normalidade".
        contract: O Contrato Estatístico (identifica colunas numéricas/categóricas).
        strangeness_std: Nº de desvios padrão acima da média que define
            o limiar de estranheza.
        rng_seed: Semente aleatória, para reprodutibilidade do treino.

    Returns:
        Um :class:`AutoencoderScorer` pronto a avaliar linhas candidatas.

    Raises:
        InsufficientDataError: Se `seed` tiver menos de `MIN_SEED_SIZE_AUTOENCODER` linhas.
    """
    if len(seed) < MIN_SEED_SIZE_AUTOENCODER:
        raise InsufficientDataError(
            f"semente com {len(seed)} linha(s) é insuficiente para treinar o "
            f"autoencoder de forma estável (mínimo: {MIN_SEED_SIZE_AUTOENCODER})"
        )

    numeric_cols = [
        c for c in seed.columns if isinstance(contract.column_stats[c], ContinuousStatistics)
    ]
    categorical_cols = [c for c in seed.columns if c not in numeric_cols]
    num_means = {c: contract.column_stats[c].media for c in numeric_cols}
    num_stds = {c: max(contract.column_stats[c].desvio, 1e-9) for c in numeric_cols}
    cat_encoders = {
        c: {categoria: i for i, categoria in enumerate(contract.column_stats[c].frequencias)}
        for c in categorical_cols
    }

    scorer_provisorio = AutoencoderScorer(
        model=None,  # type: ignore[arg-type]
        cat_encoders=cat_encoders,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        num_means=num_means,
        num_stds=num_stds,
        baseline=AutoencoderBaseline(media_perda=0.0, desvio_perda=1.0, limiar=0.0),
    )
    x = scorer_provisorio._encode([row for _, row in seed.iterrows()])

    dim_entrada = x.shape[1]
    gargalo = max(1, dim_entrada // 2)
    melhor_modelo: MLPRegressor | None = None
    melhor_perda_media = np.inf
    for tentativa in range(AUTOENCODER_TRAINING_RESTARTS):
        semente_tentativa = None if rng_seed is None else rng_seed + tentativa * 7919
        candidato = MLPRegressor(
            hidden_layer_sizes=(gargalo,),
            max_iter=AUTOENCODER_MAX_ITER,
            random_state=semente_tentativa,
        )
        candidato.fit(x, x)
        perda_media_candidato = float(np.mean((x - candidato.predict(x)) ** 2))
        if perda_media_candidato < melhor_perda_media:
            melhor_perda_media = perda_media_candidato
            melhor_modelo = candidato
    modelo = melhor_modelo

    perdas = np.mean((x - modelo.predict(x)) ** 2, axis=1)
    media = float(perdas.mean())
    desvio = max(float(perdas.std()), MIN_LOSS_STD_FLOOR)
    baseline = AutoencoderBaseline(
        media_perda=media, desvio_perda=desvio, limiar=media + strangeness_std * desvio
    )

    return AutoencoderScorer(
        modelo, cat_encoders, numeric_cols, categorical_cols, num_means, num_stds, baseline
    )


# --------------------------------------------------------------------------
# Memória vetorial (mesma política de teto do Módulo 3)
# --------------------------------------------------------------------------


def _append_to_memory(
    memory: pd.DataFrame,
    weights: np.ndarray,
    row: dict[str, Any],
    max_memory_size: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Adiciona `row` à memória, com teto via reservoir sampling (ver `MAX_MEMORY_SIZE`)."""
    if len(memory) < max_memory_size:
        nova_memoria = pd.concat([memory, pd.DataFrame([row])], ignore_index=True)
        novos_pesos = np.append(weights, 1.0)
        return nova_memoria, novos_pesos

    indice_substituir = int(rng.integers(0, max_memory_size))
    nova_memoria = memory.copy()
    nova_memoria.iloc[indice_substituir] = row
    novos_pesos = weights.copy()
    novos_pesos[indice_substituir] = 1.0
    return nova_memoria, novos_pesos


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def run_nucleo(
    seed: pd.DataFrame,
    contract: StatisticalContract,
    n_rows: int,
    dag: CausalDAG | None = None,
    initial_weights: np.ndarray | None = None,
    k_range: tuple[int, int] = (K_MIN, K_MAX),
    mahalanobis_threshold: float = DEFAULT_MAHALANOBIS_THRESHOLD,
    plausibility_threshold: float = DEFAULT_PLAUSIBILITY_THRESHOLD,
    strangeness_std: float = DEFAULT_STRANGENESS_STD,
    asp_penalty: float = DEFAULT_ASP_PENALTY,
    artifact_penalty: float = DEFAULT_ARTIFACT_PENALTY,
    autoencoder_update_interval: int = DEFAULT_AUTOENCODER_UPDATE_INTERVAL,
    max_attempts_per_row: int = MAX_ATTEMPTS_PER_ROW,
    max_memory_size: int = MAX_MEMORY_SIZE,
    rng_seed: int | None = None,
) -> NucleoOutput:
    """Expande a semente através do ciclo neural-simbólico completo (Módulo 5).

    Cada linha quimera candidata (gerada exatamente como no Módulo 3)
    passa, em sequência, pelos quatro filtros: distância de Mahalanobis,
    score de plausibilidade, validação lógica estrita (ASP, Módulo 4) e
    perda de reconstrução do autoencoder. Qualquer rejeição penaliza de
    imediato as fontes que contribuíram para a linha rejeitada — a
    aprendizagem acontece dentro do próprio lote.

    Args:
        seed: A semente do Módulo 2 (ou a memória acumulada de um ciclo anterior).
        contract: O Contrato Estatístico do Módulo 1.
        n_rows: Número de novas linhas a gerar e aprovar neste lote.
        dag: DAG opcional do Módulo 2 (agrupamento de colunas causalmente ligadas).
        initial_weights: Pesos de seleção iniciais, alinhados com `seed`
            — permite encadear "ciclos" sucessivos preservando a
            atenção negativa aprendida num ciclo anterior (ver
            `NucleoOutput.pesos_finais`). Por omissão, pesos uniformes.
        k_range, mahalanobis_threshold, plausibility_threshold,
            max_attempts_per_row, max_memory_size: ver `expansor.run_expansor`.
        strangeness_std: Nº de desvios padrão que define o limiar de estranheza.
        asp_penalty: Fator de penalização aplicado a uma rejeição lógica.
        artifact_penalty: Fator de penalização aplicado a um artefacto.
        autoencoder_update_interval: Nº de linhas aprovadas entre reciclagens do autoencoder.
        rng_seed: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        :class:`NucleoOutput` com o lote aprovado, o relatório, os
        anti-exemplos e o estado (memória + pesos) para um ciclo seguinte.

    Raises:
        InsufficientDataError: Se a semente for pequena demais para o autoencoder.
        NucleoValidationError: Se `n_rows` ou `k_range` forem inválidos.
    """
    if len(seed) < max(k_range[0], MIN_SEED_SIZE_AUTOENCODER):
        raise InsufficientDataError(
            f"semente com {len(seed)} linha(s) é insuficiente (mínimo: "
            f"{max(k_range[0], MIN_SEED_SIZE_AUTOENCODER)})"
        )
    if n_rows <= 0:
        raise NucleoValidationError(f"n_rows deve ser positivo (recebido: {n_rows})")
    if not (1 <= k_range[0] <= k_range[1]):
        raise NucleoValidationError(f"k_range inválido: {k_range}")

    warnings: list[str] = []
    rng = np.random.default_rng(rng_seed)

    mean, inv_cov, numeric_cols = build_target_distribution(contract)
    plaus_scorer = train_plausibility_model(seed, contract, rng, rng_seed)
    autoenc_scorer = train_autoencoder(seed, contract, strangeness_std, rng_seed)
    column_groups = dag_column_groups(dag, list(contract.column_stats))

    memory = seed.reset_index(drop=True).copy()
    weights = np.ones(len(memory)) if initial_weights is None else np.asarray(initial_weights, dtype=float).copy()
    if len(weights) != len(memory):
        raise NucleoValidationError(
            f"initial_weights tem {len(weights)} elemento(s), esperado {len(memory)} "
            "(um por linha da semente)"
        )

    accepted_rows: list[dict[str, Any]] = []
    recon_losses: list[float] = []
    anti_exemplos: list[AntiExample] = []
    rejeicoes = {motivo.value: 0 for motivo in NucleoRejectionReason}
    tentativas_totais = 0
    desde_ultima_reciclagem = 0
    reciclagens = 0
    orcamento_global = n_rows * GLOBAL_BUDGET_MULTIPLIER

    while len(accepted_rows) < n_rows and tentativas_totais < orcamento_global:
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        aceite_nesta_rodada = False

        for _ in range(max_attempts_per_row):
            linha, indices_usados = generate_chimera_row(memory, weights, k, rng, column_groups)
            tentativas_totais += 1

            distancia = mahalanobis_distance(linha, mean, inv_cov, numeric_cols)
            if distancia > mahalanobis_threshold:
                rejeicoes[NucleoRejectionReason.MAHALANOBIS.value] += 1
                weights = penalize_sources(weights, indices_usados, DEFAULT_INTERNAL_PENALTY)
                continue

            score = plaus_scorer.score_row(linha)
            if score < plausibility_threshold:
                rejeicoes[NucleoRejectionReason.PLAUSIBILIDADE.value] += 1
                weights = penalize_sources(weights, indices_usados, DEFAULT_INTERNAL_PENALTY)
                continue

            provas = validate_row(linha, contract.rules)
            if provas:
                rejeicoes[NucleoRejectionReason.ASP.value] += 1
                weights = penalize_sources(weights, indices_usados, asp_penalty)
                continue

            perda = autoenc_scorer.reconstruction_loss(linha)
            if perda > autoenc_scorer.baseline.limiar:
                rejeicoes[NucleoRejectionReason.ARTEFACTO.value] += 1
                weights = penalize_sources(weights, indices_usados, artifact_penalty)
                anti_exemplos.append(
                    AntiExample(
                        valores=linha,
                        perda_reconstrucao=perda,
                        limiar=autoenc_scorer.baseline.limiar,
                    )
                )
                continue

            accepted_rows.append(linha)
            recon_losses.append(perda)
            memory, weights = _append_to_memory(memory, weights, linha, max_memory_size, rng)
            desde_ultima_reciclagem += 1
            if desde_ultima_reciclagem >= autoencoder_update_interval:
                autoenc_scorer.partial_update(pd.DataFrame(accepted_rows[-autoencoder_update_interval:]))
                desde_ultima_reciclagem = 0
                reciclagens += 1
            aceite_nesta_rodada = True
            break

        if not aceite_nesta_rodada:
            rejeicoes[NucleoRejectionReason.TENTATIVAS_ESGOTADAS.value] += 1

    if len(accepted_rows) < n_rows:
        warnings.append(
            f"apenas {len(accepted_rows)}/{n_rows} linhas aprovadas dentro do "
            f"orçamento de {orcamento_global} tentativas — considere relaxar "
            "os limiares de filtragem ou fornecer uma semente maior/mais diversa"
        )

    colunas_finais = list(contract.column_stats)
    lote_df = (
        pd.DataFrame(accepted_rows)[colunas_finais]
        if accepted_rows
        else pd.DataFrame(columns=colunas_finais)
    )

    total_testado = tentativas_totais
    relatorio = NucleoBatchReport(
        linhas_solicitadas=n_rows,
        linhas_geradas=len(accepted_rows),
        tentativas_totais=tentativas_totais,
        rejeicoes_por_motivo=rejeicoes,
        taxa_rejeicao=(
            (total_testado - len(accepted_rows)) / total_testado if total_testado else 0.0
        ),
        artefactos_detectados=rejeicoes[NucleoRejectionReason.ARTEFACTO.value],
        perda_reconstrucao_media_aprovadas=float(np.mean(recon_losses)) if recon_losses else 0.0,
        reciclagens_autoencoder=reciclagens,
    )

    logger.info(
        "Núcleo Cognitivo processou lote: solicitadas=%d, geradas=%d, "
        "tentativas=%d, artefactos=%d, taxa_rejeicao=%.1f%%",
        n_rows,
        len(accepted_rows),
        tentativas_totais,
        relatorio.artefactos_detectados,
        relatorio.taxa_rejeicao * 100,
    )

    return NucleoOutput(
        lote=lote_df,
        relatorio=relatorio,
        anti_exemplos=anti_exemplos,
        memoria_final=memory,
        pesos_finais=list(weights),
        warnings=warnings,
    )
