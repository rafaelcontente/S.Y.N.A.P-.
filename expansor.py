"""Neocórtex Gerador com Atenção Dinâmica e Rejeição de Estranheza (Expansor).

Módulo 3 do S.Y.N.A.P. Expande a semente do Módulo 2 por remistura
composicional ("linhas quimera": cada coluna é copiada de uma de `K`
linhas-fonte selecionadas aleatoriamente de uma memória vetorial em
RAM), filtrando em tempo real por (1) distância de Mahalanobis face à
distribuição-alvo do Contrato e (2) um score de plausibilidade
estimado por uma floresta aleatória treinada na semente. Sinais de
atenção negativa (do Módulo 5, ou gerados internamente a partir das
próprias rejeições do lote) reduzem dinamicamente o peso de seleção
de fontes recorrentemente problemáticas.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from synap.hipocampo.hipocampo import build_correlation_matrix
from synap.hipocampo.models import CausalDAG
from synap.neocortex.models import ContinuousStatistics, StatisticalContract

from .exceptions import ExpansorValidationError, InsufficientMemoryError
from .models import (
    BatchGenerationReport,
    ExpansorOutput,
    NegativeAttentionSignal,
    RejectionReason,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

K_MIN: int = 3
K_MAX: int = 5

DEFAULT_MAHALANOBIS_THRESHOLD: float = 3.0
DEFAULT_PLAUSIBILITY_THRESHOLD: float = 0.1
MAX_ATTEMPTS_PER_ROW: int = 3
"""Tentativas de remistura por linha antes de a considerar 'tentativas esgotadas'."""

GLOBAL_BUDGET_MULTIPLIER: int = 50
"""Orçamento total de tentativas = n_rows * este fator, evita ciclos sem fim
quando os limiares são demasiado restritivos para a semente fornecida."""

DEFAULT_INTERNAL_PENALTY: float = 0.85
"""Fator de penalização aplicado às fontes de uma linha rejeitada
(atenção negativa interna, em tempo real, dentro do próprio lote)."""

MIN_SOURCE_WEIGHT: float = 1e-6
"""Piso do peso de seleção de uma fonte, evita exclusão numérica total
que impediria `numpy.random.Generator.choice` de normalizar probabilidades."""

MAX_MEMORY_SIZE: int = 50_000
"""Teto da memória vetorial em RAM. Acima deste tamanho, novas linhas
aprovadas substituem entradas aleatórias existentes (reservoir sampling)
em vez de crescerem a memória indefinidamente — necessário para escalar
a milhões de linhas geradas sem esgotar RAM ou degradar o custo de
amostragem por lote."""

PLAUSIBILITY_MODEL_TREES: int = 100
PLAUSIBILITY_MODEL_MAX_DEPTH: int = 8


# --------------------------------------------------------------------------
# Distribuição-alvo (média + covariância) a partir do Contrato
# --------------------------------------------------------------------------


def build_target_distribution(
    contract: StatisticalContract,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reconstrói a média e a matriz de covariância-alvo (colunas numéricas).

    A covariância é derivada de `desvio_i * desvio_j * correlacao_ij`,
    reutilizando a matriz de correlação do Módulo 2.

    Returns:
        Tuplo (vetor_media, matriz_covariancia_inversa, colunas_numericas).
        Se não existirem colunas numéricas no Contrato, devolve arrays vazios.
    """
    numeric_cols = [
        nome
        for nome, stat in contract.column_stats.items()
        if isinstance(stat, ContinuousStatistics)
    ]
    if not numeric_cols:
        return np.array([]), np.zeros((0, 0)), []

    mean = np.array([contract.column_stats[c].media for c in numeric_cols])
    std = np.array([contract.column_stats[c].desvio for c in numeric_cols])
    corr = build_correlation_matrix(contract).reindex(index=numeric_cols, columns=numeric_cols)
    cov = corr.to_numpy() * np.outer(std, std)
    inv_cov = np.linalg.pinv(cov)
    return mean, inv_cov, numeric_cols


def mahalanobis_distance(
    row: dict[str, Any], mean: np.ndarray, inv_cov: np.ndarray, numeric_cols: list[str]
) -> float:
    """Calcula a distância de Mahalanobis de `row` face à distribuição-alvo.

    Devolve 0.0 se não existirem colunas numéricas (filtro inaplicável).
    """
    if not numeric_cols:
        return 0.0
    x = np.array([float(row[c]) for c in numeric_cols]) - mean
    d2 = float(x @ inv_cov @ x)
    return float(np.sqrt(max(d2, 0.0)))


# --------------------------------------------------------------------------
# Modelo de plausibilidade (floresta aleatória treinada na semente)
# --------------------------------------------------------------------------


class _PlausibilityScorer:
    """Estima P(a combinação de valores existir), treinado na semente.

    Treina um classificador (floresta aleatória) para distinguir linhas
    reais da semente (classe 1) de linhas "embaralhadas" — cada coluna
    permutada independentemente, preservando as distribuições marginais
    mas destruindo a estrutura de dependência conjunta (classe 0). A
    probabilidade prevista para a classe 1 serve como score de
    plausibilidade: combinações que quebram a dependência conjunta
    aprendida tendem a parecer-se com a classe 0 e recebem score baixo.
    """

    def __init__(
        self,
        model: RandomForestClassifier,
        cat_encoders: dict[str, dict[Any, int]],
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> None:
        self._model = model
        self._cat_encoders = cat_encoders
        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols
        self._positive_class_index: int | None = (
            list(model.classes_).index(1) if 1 in model.classes_ else None
        )

    def _encode(self, rows: list[dict[str, Any]]) -> np.ndarray:
        partes = []
        if self._numeric_cols:
            partes.append(np.array([[float(r[c]) for c in self._numeric_cols] for r in rows]))
        for coluna in self._categorical_cols:
            encoder = self._cat_encoders[coluna]
            codigos = np.array([[encoder.get(r[coluna], -1)] for r in rows], dtype=float)
            partes.append(codigos)
        return np.hstack(partes) if partes else np.zeros((len(rows), 0))

    def score_row(self, row: dict[str, Any]) -> float:
        """Devolve o score de plausibilidade (probabilidade de classe 1) de `row`."""
        if self._positive_class_index is None:
            # O treino não observou nenhum exemplo negativo distinguível
            # (degenerado); assume-se plausibilidade máxima.
            return 1.0
        x = self._encode([row])
        proba = self._model.predict_proba(x)[0]
        return float(proba[self._positive_class_index])


def _encode_dataframe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cat_encoders: dict[str, dict[Any, int]],
) -> np.ndarray:
    """Codifica um DataFrame para features numéricas (categóricas via mapa ordinal)."""
    partes = []
    if numeric_cols:
        partes.append(df[numeric_cols].to_numpy(dtype=float))
    for coluna in categorical_cols:
        encoder = cat_encoders[coluna]
        codigos = df[coluna].map(lambda v, enc=encoder: enc.get(v, -1)).to_numpy(dtype=float)
        partes.append(codigos.reshape(-1, 1))
    return np.hstack(partes) if partes else np.zeros((len(df), 0))


def train_plausibility_model(
    seed: pd.DataFrame,
    contract: StatisticalContract,
    rng: np.random.Generator,
    rng_seed: int | None = None,
) -> _PlausibilityScorer:
    """Treina o modelo de plausibilidade (floresta aleatória) na semente.

    Args:
        seed: A semente (ou memória atual) usada como conjunto de treino positivo.
        contract: O Contrato Estatístico (identifica colunas numéricas vs categóricas).
        rng: Gerador aleatório para a criação dos exemplos negativos (permutação).
        rng_seed: Semente para a floresta aleatória, para reprodutibilidade.

    Returns:
        Um :class:`_PlausibilityScorer` pronto a avaliar linhas candidatas.
    """
    numeric_cols = [
        c for c in seed.columns if isinstance(contract.column_stats[c], ContinuousStatistics)
    ]
    categorical_cols = [c for c in seed.columns if c not in numeric_cols]
    cat_encoders = {
        c: {categoria: i for i, categoria in enumerate(contract.column_stats[c].frequencias)}
        for c in categorical_cols
    }

    x_positivo = _encode_dataframe(seed, numeric_cols, categorical_cols, cat_encoders)

    negativo_df = seed.copy()
    for coluna in negativo_df.columns:
        negativo_df[coluna] = rng.permutation(negativo_df[coluna].to_numpy())
    x_negativo = _encode_dataframe(negativo_df, numeric_cols, categorical_cols, cat_encoders)

    x = np.vstack([x_positivo, x_negativo])
    y = np.concatenate([np.ones(len(x_positivo)), np.zeros(len(x_negativo))])

    modelo = RandomForestClassifier(
        n_estimators=PLAUSIBILITY_MODEL_TREES,
        max_depth=PLAUSIBILITY_MODEL_MAX_DEPTH,
        random_state=rng_seed,
        n_jobs=-1,
    )
    modelo.fit(x, y)
    return _PlausibilityScorer(modelo, cat_encoders, numeric_cols, categorical_cols)


# --------------------------------------------------------------------------
# Grupos de colunas (opcional, a partir da DAG do Módulo 2)
# --------------------------------------------------------------------------


def dag_column_groups(dag: CausalDAG | None, all_columns: list[str]) -> list[list[str]] | None:
    """Agrupa colunas causalmente ligadas na DAG, para serem copiadas em bloco.

    Colunas dentro do mesmo componente conexo (não-dirigido) da DAG são
    tratadas como um grupo indivisível durante a remistura — todo o
    grupo é copiado de uma única linha-fonte, preservando a coerência
    interna de clusters causalmente ligados. Sem uma DAG, cada coluna
    é remisturada de forma totalmente independente.

    Args:
        dag: A DAG do Módulo 2, ou `None` para remistura por coluna.
        all_columns: Todas as colunas do Contrato (garante cobertura total).

    Returns:
        Lista de grupos de nomes de coluna, ou `None` se `dag` for `None`.
    """
    if dag is None:
        return None
    grafo = nx.Graph()
    grafo.add_nodes_from(dag.nos)
    grafo.add_edges_from((e.origem, e.destino) for e in dag.arestas)
    grupos = [list(componente) for componente in nx.connected_components(grafo)]
    colunas_cobertas = {c for grupo in grupos for c in grupo}
    for coluna in all_columns:
        if coluna not in colunas_cobertas:
            grupos.append([coluna])
    return grupos


# --------------------------------------------------------------------------
# Remistura composicional ("linha quimera")
# --------------------------------------------------------------------------


def generate_chimera_row(
    memory: pd.DataFrame,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    column_groups: list[list[str]] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    """Gera uma linha quimera combinando colunas de `k` linhas-fonte da memória.

    Args:
        memory: A memória vetorial atual (todas as linhas aprovadas até agora).
        weights: Pesos de seleção (não-normalizados) por linha da memória.
        k: Número de linhas-fonte a selecionar.
        rng: Gerador aleatório.
        column_groups: Grupos de colunas a copiar em bloco (ver :func:`dag_column_groups`).
            Se `None`, cada coluna é remisturada independentemente.

    Returns:
        Tuplo (linha_gerada, indices_fonte_utilizados) — os índices das
        linhas da memória que efetivamente contribuíram com pelo menos
        uma coluna (subconjunto das `k` selecionadas).
    """
    pesos_positivos = np.clip(weights, MIN_SOURCE_WEIGHT, None)
    probs = pesos_positivos / pesos_positivos.sum()
    k_efetivo = max(1, min(k, len(memory)))
    fontes_candidatas = rng.choice(len(memory), size=k_efetivo, replace=False, p=probs)

    grupos = column_groups if column_groups is not None else [[c] for c in memory.columns]
    linha: dict[str, Any] = {}
    usados: set[int] = set()
    for grupo in grupos:
        fonte = int(rng.choice(fontes_candidatas))
        usados.add(fonte)
        origem = memory.iloc[fonte]
        for coluna in grupo:
            if coluna in memory.columns:
                linha[coluna] = origem[coluna]
    return linha, sorted(usados)


def penalize_sources(weights: np.ndarray, indices: list[int], factor: float) -> np.ndarray:
    """Aplica um fator de penalização multiplicativo aos pesos das fontes indicadas.

    Args:
        weights: Pesos de seleção atuais.
        indices: Índices das linhas-fonte a penalizar.
        factor: Fator multiplicativo (0 < factor <= 1).

    Returns:
        Novo array de pesos (não muta o array recebido).
    """
    novos_pesos = weights.copy()
    for indice in indices:
        if 0 <= indice < len(novos_pesos):
            novos_pesos[indice] = max(novos_pesos[indice] * factor, MIN_SOURCE_WEIGHT)
    return novos_pesos


def apply_negative_attention(
    weights: np.ndarray, signals: list[NegativeAttentionSignal]
) -> tuple[np.ndarray, list[str]]:
    """Aplica os sinais de atenção negativa do Módulo 5 aos pesos iniciais do lote.

    Returns:
        Tuplo (pesos_ajustados, avisos) — um aviso é gerado por cada
        sinal que refira um índice fora do intervalo da memória atual.
    """
    novos_pesos = weights.copy()
    warnings: list[str] = []
    for sinal in signals:
        if 0 <= sinal.indice_fonte < len(novos_pesos):
            novos_pesos[sinal.indice_fonte] = max(
                novos_pesos[sinal.indice_fonte] * sinal.penalizacao, MIN_SOURCE_WEIGHT
            )
        else:
            warnings.append(
                f"sinal de atenção negativa refere índice fora da memória "
                f"atual ({sinal.indice_fonte}) — ignorado"
            )
    return novos_pesos, warnings


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


def run_expansor(
    seed: pd.DataFrame,
    contract: StatisticalContract,
    n_rows: int,
    dag: CausalDAG | None = None,
    k_range: tuple[int, int] = (K_MIN, K_MAX),
    mahalanobis_threshold: float = DEFAULT_MAHALANOBIS_THRESHOLD,
    plausibility_threshold: float = DEFAULT_PLAUSIBILITY_THRESHOLD,
    negative_attention: list[NegativeAttentionSignal] | None = None,
    max_attempts_per_row: int = MAX_ATTEMPTS_PER_ROW,
    internal_penalty: float = DEFAULT_INTERNAL_PENALTY,
    max_memory_size: int = MAX_MEMORY_SIZE,
    rng_seed: int | None = None,
) -> ExpansorOutput:
    """Expande a semente por remistura composicional filtrada (Módulo 3 do S.Y.N.A.P.).

    Args:
        seed: A semente do Módulo 2 (ou a memória acumulada de um lote anterior).
        contract: O Contrato Estatístico do Módulo 1.
        n_rows: Número de novas linhas a gerar e aprovar neste lote.
        dag: DAG opcional do Módulo 2 — se fornecida, colunas causalmente
            ligadas são copiadas em bloco da mesma linha-fonte (ver
            :func:`dag_column_groups`).
        k_range: Intervalo (mín, máx) do número de linhas-fonte por linha quimera.
        mahalanobis_threshold: Distância de Mahalanobis máxima aceite.
        plausibility_threshold: Score de plausibilidade mínimo aceite.
        negative_attention: Sinais de atenção negativa do Módulo 5, aplicados
            aos pesos de seleção antes do início deste lote.
        max_attempts_per_row: Tentativas de remistura por linha antes de a
            reportar como "tentativas esgotadas".
        internal_penalty: Fator de penalização aplicado, em tempo real,
            às fontes de uma linha rejeitada dentro do próprio lote.
        max_memory_size: Teto da memória vetorial (ver `MAX_MEMORY_SIZE`).
        rng_seed: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        :class:`ExpansorOutput` com o lote aprovado, o relatório de
        filtragem e o novo tamanho da memória vetorial.

    Raises:
        InsufficientMemoryError: Se a semente tiver menos linhas do que `k_range[0]`.
        ExpansorValidationError: Se `n_rows` ou `k_range` forem inválidos.
    """
    if len(seed) < k_range[0]:
        raise InsufficientMemoryError(
            f"semente com {len(seed)} linha(s) é insuficiente para remistura "
            f"com K mínimo de {k_range[0]} fontes"
        )
    if n_rows <= 0:
        raise ExpansorValidationError(f"n_rows deve ser positivo (recebido: {n_rows})")
    if not (1 <= k_range[0] <= k_range[1]):
        raise ExpansorValidationError(f"k_range inválido: {k_range}")

    warnings: list[str] = []
    rng = np.random.default_rng(rng_seed)

    mean, inv_cov, numeric_cols = build_target_distribution(contract)
    if not numeric_cols:
        warnings.append(
            "Contrato sem colunas numéricas — filtro de distância de "
            "Mahalanobis desativado (distância sempre 0.0)"
        )

    scorer = train_plausibility_model(seed, contract, rng, rng_seed)
    column_groups = dag_column_groups(dag, list(contract.column_stats))

    memory = seed.reset_index(drop=True).copy()
    weights = np.ones(len(memory))
    if negative_attention:
        weights, warnings_atencao = apply_negative_attention(weights, negative_attention)
        warnings.extend(warnings_atencao)

    accepted_rows: list[dict[str, Any]] = []
    distancias: list[float] = []
    scores: list[float] = []
    rejeicoes = {motivo.value: 0 for motivo in RejectionReason}
    tentativas_totais = 0
    orcamento_global = n_rows * GLOBAL_BUDGET_MULTIPLIER

    while len(accepted_rows) < n_rows and tentativas_totais < orcamento_global:
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        aceite_nesta_rodada = False

        for _ in range(max_attempts_per_row):
            linha, indices_usados = generate_chimera_row(memory, weights, k, rng, column_groups)
            tentativas_totais += 1

            distancia = mahalanobis_distance(linha, mean, inv_cov, numeric_cols)
            if distancia > mahalanobis_threshold:
                rejeicoes[RejectionReason.MAHALANOBIS.value] += 1
                weights = penalize_sources(weights, indices_usados, internal_penalty)
                continue

            score = scorer.score_row(linha)
            if score < plausibility_threshold:
                rejeicoes[RejectionReason.PLAUSIBILIDADE.value] += 1
                weights = penalize_sources(weights, indices_usados, internal_penalty)
                continue

            accepted_rows.append(linha)
            distancias.append(distancia)
            scores.append(score)
            memory, weights = _append_to_memory(memory, weights, linha, max_memory_size, rng)
            aceite_nesta_rodada = True
            break

        if not aceite_nesta_rodada:
            rejeicoes[RejectionReason.TENTATIVAS_ESGOTADAS.value] += 1

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
    relatorio = BatchGenerationReport(
        linhas_solicitadas=n_rows,
        linhas_geradas=len(accepted_rows),
        tentativas_totais=tentativas_totais,
        rejeicoes_por_motivo=rejeicoes,
        taxa_rejeicao=(
            (total_testado - len(accepted_rows)) / total_testado if total_testado else 0.0
        ),
        distancia_mahalanobis_media_aprovadas=float(np.mean(distancias)) if distancias else 0.0,
        score_plausibilidade_medio_aprovadas=float(np.mean(scores)) if scores else 0.0,
    )

    logger.info(
        "Lote do Expansor gerado: solicitadas=%d, geradas=%d, tentativas=%d, "
        "taxa_rejeicao=%.1f%%",
        n_rows,
        len(accepted_rows),
        tentativas_totais,
        relatorio.taxa_rejeicao * 100,
    )

    return ExpansorOutput(
        lote=lote_df,
        relatorio=relatorio,
        tamanho_memoria_final=len(memory),
        warnings=warnings,
    )
