"""Auditor de Utilidade, Privacidade e Valor Prático (Provas de Disrupção).

Módulo 8 do S.Y.N.A.P. Prova três coisas que o Módulo 7, sozinho, não
prova: (1) **utilidade estatística** generalizada via TSTR/TRTR em
várias tarefas-alvo, não apenas uma; (2) **segurança de privacidade**
— Distance to Closest Record, Nearest-Neighbour Distance Ratio,
correspondência exacta e um Ataque de Inferência de Pertença simples,
que juntos detectam memorização que a fidelidade estatística por si só
nunca revelaria; (3) **valor prático**, consolidado num Relatório de
Prontidão legível por um não-especialista.

A privacidade tem poder de veto: um dataset pode ter utilidade e
fidelidade excelentes e ainda assim ser bloqueado se memorizar dados reais.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from synap.compilador.models import TargetType
from synap.neocortex.models import CategoricalStatistics, ContinuousStatistics, StatisticalContract

from .exceptions import AuditorValidationError, InsufficientHoldoutError
from .models import (
    AuditorOutput,
    DCRAssessment,
    ExactMatchAssessment,
    MembershipInferenceAssessment,
    NNDRAssessment,
    PrivacyAssessment,
    UtilityAssessment,
    UtilityTaskResult,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

DEFAULT_UTILITY_RETENTION_THRESHOLD: float = 0.85
DEFAULT_DCR_RATIO_THRESHOLD: float = 0.5
"""Se a mediana do DCR sintético->real cair abaixo desta fração da
mediana real->real, considera-se risco de memorização."""

DEFAULT_MIA_AUC_TOLERANCE: float = 0.10
"""O atacante de inferência de pertença é considerado "seguro" se a
sua AUC estiver dentro de 0.5 ± esta tolerância."""

MIN_HOLDOUT_SIZE: int = 50
MAX_TARGET_TASKS: int = 3
MAX_DISTANCE_SAMPLE: int = 2000
"""Teto de linhas amostradas para os cálculos de distância (DCR/NNDR/MIA)
— O(n·m) por construção; acima disto, amostra-se para manter o custo controlado."""

EXACT_MATCH_ROUNDING_DECIMALS: int = 6
"""Precisão usada para comparar colunas numéricas na deteção de
correspondência exacta. Deliberadamente ALTA (não uma tolerância
larga): cópias genuínas de linhas reais preservam a precisão total do
valor original, pelo que não precisam de tolerância; um arredondamento
grosseiro (ex.: 2 casas decimais) produziria colisões por acaso em
dados contínuos independentes, gerando falsos positivos."""


# --------------------------------------------------------------------------
# Seleção de colunas-alvo e codificação (consistente com o resto do sistema)
# --------------------------------------------------------------------------


def select_target_columns(
    contract: StatisticalContract, target_columns: list[str] | None = None, max_tasks: int = MAX_TARGET_TASKS
) -> list[str]:
    """Seleciona até `max_tasks` colunas-alvo para a avaliação de utilidade (TSTR/TRTR).

    Sem `target_columns` explícitas, prioriza colunas categóricas
    (tarefas de classificação, mais informativas para utilidade de
    negócio) e completa com as numéricas de maior variância.
    """
    if target_columns is not None:
        desconhecidas = [c for c in target_columns if c not in contract.column_stats]
        if desconhecidas:
            raise AuditorValidationError(f"coluna(s)-alvo inexistente(s) no Contrato: {desconhecidas}")
        return target_columns[:max_tasks]

    categoricas = [c for c, s in contract.column_stats.items() if isinstance(s, CategoricalStatistics)]
    numericas = sorted(
        (c for c, s in contract.column_stats.items() if isinstance(s, ContinuousStatistics)),
        key=lambda c: contract.column_stats[c].desvio,
        reverse=True,
    )
    candidatas = categoricas + numericas
    if not candidatas:
        raise AuditorValidationError("o Contrato não declara nenhuma coluna utilizável como alvo")
    return candidatas[:max_tasks]


def _encode_classifier_features(
    df: pd.DataFrame, contract: StatisticalContract, exclude_col: str | None = None
) -> np.ndarray:
    """Codifica colunas (exceto `exclude_col`) para features de modelo: z-score
    numérico (média/desvio do Contrato) + ordinal categórico."""
    colunas = [c for c in df.columns if c != exclude_col]
    partes = []
    for coluna in colunas:
        stat = contract.column_stats[coluna]
        if isinstance(stat, ContinuousStatistics):
            desvio = max(stat.desvio, 1e-9)
            partes.append(((df[coluna].to_numpy(dtype=float) - stat.media) / desvio).reshape(-1, 1))
        else:
            encoder = {categoria: i for i, categoria in enumerate(stat.frequencias)}
            codigos = df[coluna].map(lambda v, enc=encoder: enc.get(v, -1)).to_numpy(dtype=float)
            partes.append(codigos.reshape(-1, 1))
    return np.hstack(partes) if partes else np.zeros((len(df), 0))


def _encode_for_distance(df: pd.DataFrame, contract: StatisticalContract) -> np.ndarray:
    """Codifica TODAS as colunas para cálculos de distância (DCR/NNDR/MIA).

    Categóricas são normalizadas para uma escala comparável à dos
    z-scores numéricos (`código / (nº categorias - 1)`), para que uma
    diferença categórica não domine nem seja irrelevante face às
    diferenças numéricas na distância Euclidiana.
    """
    partes = []
    for coluna in df.columns:
        stat = contract.column_stats[coluna]
        if isinstance(stat, ContinuousStatistics):
            desvio = max(stat.desvio, 1e-9)
            partes.append(((df[coluna].to_numpy(dtype=float) - stat.media) / desvio).reshape(-1, 1))
        else:
            categorias = list(stat.frequencias)
            encoder = {categoria: i for i, categoria in enumerate(categorias)}
            escala = max(len(categorias) - 1, 1)
            codigos = df[coluna].map(lambda v, enc=encoder: enc.get(v, -1)).to_numpy(dtype=float) / escala
            partes.append(codigos.reshape(-1, 1))
    return np.hstack(partes) if partes else np.zeros((len(df), 0))


def _train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    contract: StatisticalContract,
    target_column: str,
    rng_seed: int | None,
) -> tuple[str, float, TargetType]:
    """Treina em `train_df` e avalia em `test_df` — o bloco reutilizável do TSTR/TRTR."""
    is_categorico = isinstance(contract.column_stats[target_column], CategoricalStatistics)
    x_treino = _encode_classifier_features(train_df, contract, target_column)
    y_treino = train_df[target_column].to_numpy()
    x_teste = _encode_classifier_features(test_df, contract, target_column)
    y_teste = test_df[target_column].to_numpy()

    if is_categorico:
        if len(np.unique(y_treino)) < 2:
            raise AuditorValidationError(
                f"treino com uma única classe em '{target_column}' — impossível treinar classificador"
            )
        modelo = LogisticRegression(max_iter=1000, random_state=rng_seed)
        modelo.fit(x_treino, y_treino)
        proba = modelo.predict_proba(x_teste)
        if len(modelo.classes_) == 2:
            valor = float(roc_auc_score(y_teste, proba[:, 1]))
        else:
            valor = float(roc_auc_score(y_teste, proba, multi_class="ovr", average="macro", labels=modelo.classes_))
        return "AUC", valor, TargetType.CATEGORICO

    modelo = LinearRegression()
    modelo.fit(x_treino, y_treino)
    previsao = modelo.predict(x_teste)
    rmse = float(np.sqrt(mean_squared_error(y_teste, previsao)))
    return "RMSE", rmse, TargetType.NUMERICO


def _retention_from_metrics(metric_name: str, tstr: float, trtr: float) -> float:
    """Converte TSTR/TRTR numa retenção de utilidade em [0, 1] (1.0 = idêntico ao real)."""
    if metric_name == "AUC":
        if trtr <= 0:
            return 0.0
        return float(np.clip(tstr / trtr, 0.0, 1.0))
    if tstr <= 0:
        return 0.0
    return float(np.clip(trtr / tstr, 0.0, 1.0))


# --------------------------------------------------------------------------
# Pilar 1 — Utilidade estatística (TSTR/TRTR multi-tarefa)
# --------------------------------------------------------------------------


def assess_utility(
    dataset_sintetico: pd.DataFrame,
    real_holdout: pd.DataFrame,
    contract: StatisticalContract,
    target_columns: list[str] | None = None,
    test_size: float = 0.2,
    utility_threshold: float = DEFAULT_UTILITY_RETENTION_THRESHOLD,
    rng_seed: int | None = None,
) -> UtilityAssessment:
    """Avalia a utilidade estatística via TSTR vs. TRTR em várias tarefas-alvo.

    Para cada coluna-alvo: treina um modelo no SINTÉTICO e testa no
    REAL (TSTR); treina um modelo idêntico numa fração do REAL e testa
    na fração restante (TRTR, baseline); a retenção de utilidade é a
    fração do desempenho TRTR preservada pelo TSTR.
    """
    alvos = select_target_columns(contract, target_columns)
    tarefas: list[UtilityTaskResult] = []

    for alvo in alvos:
        estratificar = real_holdout[alvo] if isinstance(contract.column_stats[alvo], CategoricalStatistics) else None
        treino_real, teste_real = train_test_split(
            real_holdout, test_size=test_size, random_state=rng_seed, stratify=estratificar
        )
        metrica_nome, valor_tstr, tipo = _train_and_eval(dataset_sintetico, teste_real, contract, alvo, rng_seed)
        _, valor_trtr, _ = _train_and_eval(treino_real, teste_real, contract, alvo, rng_seed)
        retencao = _retention_from_metrics(metrica_nome, valor_tstr, valor_trtr)

        tarefas.append(
            UtilityTaskResult(
                coluna_alvo=alvo,
                tipo_alvo=tipo,
                metrica_nome=metrica_nome,
                valor_tstr=valor_tstr,
                valor_trtr=valor_trtr,
                retencao_utilidade=retencao,
            )
        )

    retencao_media = float(np.mean([t.retencao_utilidade for t in tarefas]))
    return UtilityAssessment(
        tarefas=tarefas, retencao_media=retencao_media, aprovado=retencao_media >= utility_threshold
    )


# --------------------------------------------------------------------------
# Pilar 2 — Privacidade (DCR, NNDR, correspondência exacta, MIA)
# --------------------------------------------------------------------------


def distance_to_closest_record(
    query_df: pd.DataFrame, reference_df: pd.DataFrame, contract: StatisticalContract, k: int = 1
) -> np.ndarray:
    """Distância Euclidiana (features codificadas) de cada linha de `query_df`
    à sua `k`-ésima linha mais próxima em `reference_df`."""
    x_query = _encode_for_distance(query_df, contract)
    x_ref = _encode_for_distance(reference_df, contract)
    vizinhos = NearestNeighbors(n_neighbors=min(k, len(x_ref))).fit(x_ref)
    distancias, _ = vizinhos.kneighbors(x_query)
    return distancias[:, k - 1]


def _sample(df: pd.DataFrame, max_rows: int, rng_seed: int | None) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=rng_seed)


def _split_holdout_halves(real_holdout: pd.DataFrame, rng_seed: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o holdout real em duas metades — o baseline interno real->real."""
    return train_test_split(real_holdout, test_size=0.5, random_state=rng_seed)


def dcr_assessment(
    dataset_sintetico: pd.DataFrame, real_holdout: pd.DataFrame, contract: StatisticalContract, rng_seed: int | None = None
) -> DCRAssessment:
    """Distance to Closest Record: deteta memorização por proximidade excessiva."""
    metade_a, metade_b = _split_holdout_halves(real_holdout, rng_seed)
    dcr_real_real = distance_to_closest_record(metade_a, metade_b, contract, k=1)
    mediana_real_real = float(np.median(dcr_real_real))

    amostra_sintetica = _sample(dataset_sintetico, MAX_DISTANCE_SAMPLE, rng_seed)
    dcr_sint_real = distance_to_closest_record(amostra_sintetica, real_holdout, contract, k=1)
    mediana_sint_real = float(np.median(dcr_sint_real))

    razao = mediana_sint_real / mediana_real_real if mediana_real_real > 0 else float("inf")
    limiar_suspeita = 0.1 * mediana_real_real
    linhas_suspeitas = int((dcr_sint_real < limiar_suspeita).sum())

    return DCRAssessment(
        dcr_sintetico_real_mediana=mediana_sint_real,
        dcr_real_real_mediana=mediana_real_real,
        razao=razao,
        percentil_5_sintetico_real=float(np.percentile(dcr_sint_real, 5)),
        linhas_suspeitas=linhas_suspeitas,
    )


def nndr_assessment(
    dataset_sintetico: pd.DataFrame, real_holdout: pd.DataFrame, contract: StatisticalContract, rng_seed: int | None = None
) -> NNDRAssessment:
    """Nearest-Neighbour Distance Ratio: segunda defesa contra falsos positivos do DCR."""
    metade_a, metade_b = _split_holdout_halves(real_holdout, rng_seed)

    x_a = _encode_for_distance(metade_a, contract)
    x_b = _encode_for_distance(metade_b, contract)
    viz_real = NearestNeighbors(n_neighbors=min(2, len(x_b))).fit(x_b)
    dist_real, _ = viz_real.kneighbors(x_a)
    nndr_real = dist_real[:, 0] / np.maximum(dist_real[:, -1], 1e-12)

    amostra_sintetica = _sample(dataset_sintetico, MAX_DISTANCE_SAMPLE, rng_seed)
    x_sint = _encode_for_distance(amostra_sintetica, contract)
    x_real = _encode_for_distance(real_holdout, contract)
    viz_sint = NearestNeighbors(n_neighbors=min(2, len(x_real))).fit(x_real)
    dist_sint, _ = viz_sint.kneighbors(x_sint)
    nndr_sint = dist_sint[:, 0] / np.maximum(dist_sint[:, -1], 1e-12)

    return NNDRAssessment(
        nndr_sintetico_mediana=float(np.median(nndr_sint)), nndr_real_mediana=float(np.median(nndr_real))
    )


def exact_match_assessment(
    dataset_sintetico: pd.DataFrame, real_holdout: pd.DataFrame, contract: StatisticalContract
) -> ExactMatchAssessment:
    """Verifica correspondências exactas linha-a-linha (arredondadas para leakage bruto)."""
    def _arredondar(df: pd.DataFrame) -> pd.DataFrame:
        saida = df.copy()
        for coluna in df.columns:
            if isinstance(contract.column_stats[coluna], ContinuousStatistics):
                saida[coluna] = saida[coluna].round(EXACT_MATCH_ROUNDING_DECIMALS)
        return saida

    sint_r = _arredondar(dataset_sintetico)
    real_r = _arredondar(real_holdout)
    fusao = sint_r.reset_index().merge(real_r, on=list(dataset_sintetico.columns), how="inner")
    n_correspondencias = fusao["index"].nunique()
    fracao = n_correspondencias / len(dataset_sintetico) if len(dataset_sintetico) else 0.0
    return ExactMatchAssessment(n_correspondencias_exactas=n_correspondencias, fracao=fracao)


def membership_inference_risk(
    dataset_sintetico: pd.DataFrame,
    real_holdout: pd.DataFrame,
    training_reference: pd.DataFrame | None,
    contract: StatisticalContract,
    mia_auc_tolerance: float = DEFAULT_MIA_AUC_TOLERANCE,
    rng_seed: int | None = None,
) -> MembershipInferenceAssessment:
    """Ataque de Inferência de Pertença simples, baseado em distância ao sintético.

    Um atacante treina um classificador que tenta distinguir linhas que
    ESTIVERAM na âncora real do gerador (`training_reference`, membros)
    de linhas que nunca viu (`real_holdout`, não-membros), usando como
    única feature a distância de cada linha ao sintético mais próximo
    (a intuição: se o gerador memorizou membros, estes tendem a estar
    anormalmente próximos do sintético). AUC ~ 0.5 = o ataque falha, o
    gerador não vaza informação de pertença.

    Requer `training_reference` — sem ele, este ataque não pode ser
    montado e o risco correspondente NÃO é avaliado (não fingido).
    """
    if training_reference is None or len(training_reference) < MIN_HOLDOUT_SIZE:
        return MembershipInferenceAssessment(
            avaliavel=False,
            motivo_nao_avaliavel=(
                "'training_reference' não fornecido ou insuficiente — o risco de "
                "inferência de pertença NÃO foi avaliado (dados de treino do "
                "gerador desconhecidos para o auditor)"
            ),
        )

    amostra_sintetica = _sample(dataset_sintetico, MAX_DISTANCE_SAMPLE, rng_seed)
    dist_membros = distance_to_closest_record(training_reference, amostra_sintetica, contract, k=1)
    dist_nao_membros = distance_to_closest_record(real_holdout, amostra_sintetica, contract, k=1)

    x = np.concatenate([dist_membros, dist_nao_membros]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(dist_membros)), np.zeros(len(dist_nao_membros))])

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.3, random_state=rng_seed, stratify=y
    )
    atacante = LogisticRegression(random_state=rng_seed)
    atacante.fit(x_treino, y_treino)
    proba = atacante.predict_proba(x_teste)[:, 1]
    auc = float(roc_auc_score(y_teste, proba))

    return MembershipInferenceAssessment(
        avaliavel=True, auc=auc, seguro=abs(auc - 0.5) <= mia_auc_tolerance
    )


def assess_privacy(
    dataset_sintetico: pd.DataFrame,
    real_holdout: pd.DataFrame | None,
    contract: StatisticalContract,
    training_reference: pd.DataFrame | None = None,
    dcr_ratio_threshold: float = DEFAULT_DCR_RATIO_THRESHOLD,
    mia_auc_tolerance: float = DEFAULT_MIA_AUC_TOLERANCE,
    rng_seed: int | None = None,
) -> PrivacyAssessment:
    """Avalia a privacidade em quatro frentes: DCR, NNDR, correspondência exacta e MIA.

    Se `real_holdout` for `None` ou pequeno demais, devolve
    `avaliavel=False` — o módulo nunca reivindica segurança de
    privacidade que não testou.
    """
    if real_holdout is None or len(real_holdout) < MIN_HOLDOUT_SIZE:
        return PrivacyAssessment(
            avaliavel=False,
            aprovado=False,
            motivo=(
                "sem holdout real suficiente — privacidade NÃO avaliada. "
                "Isto é, em si, um risco a reportar: não se pode reivindicar "
                "segurança de privacidade sem a testar com dados reais."
            ),
        )

    dcr = dcr_assessment(dataset_sintetico, real_holdout, contract, rng_seed)
    nndr = nndr_assessment(dataset_sintetico, real_holdout, contract, rng_seed)
    exato = exact_match_assessment(dataset_sintetico, real_holdout, contract)
    mia = membership_inference_risk(dataset_sintetico, real_holdout, training_reference, contract, mia_auc_tolerance, rng_seed)

    motivos: list[str] = []
    if dcr.razao < dcr_ratio_threshold:
        motivos.append(
            f"DCR sintético→real muito abaixo do esperado (razão={dcr.razao:.2f} < "
            f"{dcr_ratio_threshold}) — risco de memorização"
        )
    if exato.n_correspondencias_exactas > 0:
        motivos.append(
            f"{exato.n_correspondencias_exactas} correspondência(s) exacta(s) "
            "linha-a-linha detectada(s) com o holdout real"
        )
    if mia.avaliavel and not mia.seguro:
        motivos.append(f"Ataque de Inferência de Pertença bem-sucedido (AUC={mia.auc:.3f})")

    return PrivacyAssessment(
        avaliavel=True,
        dcr=dcr,
        nndr=nndr,
        correspondencia_exacta=exato,
        mia=mia,
        aprovado=not motivos,
        motivo="; ".join(motivos) if motivos else None,
    )


# --------------------------------------------------------------------------
# Relatório de Prontidão (PDF)
# --------------------------------------------------------------------------


def generate_readiness_report(
    utilidade: UtilityAssessment, privacidade: PrivacyAssessment, aprovado_global: bool, output_path: Path
) -> Path:
    """Gera o Relatório de Prontidão: veredito por pilar, legível por um não-especialista."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # --- Página 1: veredito global ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        linhas = [
            "Relatório de Prontidão — S.Y.N.A.P. (Módulo 8)",
            "",
            f"Veredito global: {'APROVADO' if aprovado_global else 'BLOQUEADO'}",
            "",
            f"Utilidade estatística (TSTR/TRTR): retenção média = {utilidade.retencao_media:.1%} "
            f"({'aprovado' if utilidade.aprovado else 'reprovado'})",
        ]
        for tarefa in utilidade.tarefas:
            linhas.append(
                f"  - {tarefa.coluna_alvo}: {tarefa.metrica_nome} sintético={tarefa.valor_tstr:.4f}, "
                f"real={tarefa.valor_trtr:.4f}, retenção={tarefa.retencao_utilidade:.1%}"
            )
        linhas.append("")
        if not privacidade.avaliavel:
            linhas.append("Privacidade: NÃO AVALIADA — " + (privacidade.motivo or ""))
        else:
            linhas.append(f"Privacidade: {'aprovado' if privacidade.aprovado else 'BLOQUEADO'}")
            if privacidade.dcr:
                linhas.append(
                    f"  - DCR sintético→real: mediana={privacidade.dcr.dcr_sintetico_real_mediana:.3f} "
                    f"(baseline real→real={privacidade.dcr.dcr_real_real_mediana:.3f}, "
                    f"razão={privacidade.dcr.razao:.2f})"
                )
            if privacidade.correspondencia_exacta:
                linhas.append(
                    f"  - Correspondências exactas: {privacidade.correspondencia_exacta.n_correspondencias_exactas}"
                )
            if privacidade.mia and privacidade.mia.avaliavel:
                linhas.append(f"  - MIA: AUC={privacidade.mia.auc:.3f}")
            elif privacidade.mia:
                linhas.append(f"  - MIA: não avaliável ({privacidade.mia.motivo_nao_avaliavel})")
            if privacidade.motivo:
                linhas.append(f"  Motivo do bloqueio: {privacidade.motivo}")

        ax.text(0.03, 0.97, "\n".join(linhas), va="top", fontsize=9, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Página 2: TSTR vs TRTR por tarefa ---
        if utilidade.tarefas:
            fig, ax = plt.subplots(figsize=(8.27, 6))
            nomes = [t.coluna_alvo for t in utilidade.tarefas]
            x_pos = np.arange(len(nomes))
            ax.bar(x_pos - 0.2, [t.valor_tstr for t in utilidade.tarefas], width=0.4, label="Sintético (TSTR)", color="#14213D")
            ax.bar(x_pos + 0.2, [t.valor_trtr for t in utilidade.tarefas], width=0.4, label="Real (TRTR)", color="#C9A35F")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(nomes)
            ax.set_title("Utilidade por tarefa: Sintético vs. Real")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

    return output_path


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def run_auditor(
    dataset_sintetico: pd.DataFrame,
    contract: StatisticalContract,
    real_holdout: pd.DataFrame,
    training_reference: pd.DataFrame | None = None,
    target_columns: list[str] | None = None,
    utility_threshold: float = DEFAULT_UTILITY_RETENTION_THRESHOLD,
    dcr_ratio_threshold: float = DEFAULT_DCR_RATIO_THRESHOLD,
    mia_auc_tolerance: float = DEFAULT_MIA_AUC_TOLERANCE,
    output_dir: Path = Path("/mnt/user-data/outputs"),
    rng_seed: int | None = None,
) -> AuditorOutput:
    """Executa a auditoria completa (Módulo 8): utilidade + privacidade + relatório.

    Args:
        dataset_sintetico: O dataset final (tipicamente a saída do Módulo 7).
        contract: O Contrato Estatístico do Módulo 1.
        real_holdout: Amostra de dados reais nunca usada para gerar o
            sintético — obrigatória (ver `InsufficientHoldoutError`).
        training_reference: Amostra dos dados reais que FORAM usados
            para ancorar o gerador (Módulo 1) — opcional, só necessária
            para o Ataque de Inferência de Pertença.
        target_columns: Colunas-alvo para TSTR/TRTR (ver :func:`select_target_columns`).
        utility_threshold: Retenção de utilidade mínima para aprovação.
        dcr_ratio_threshold: Razão DCR mínima antes de soar a alarme de memorização.
        mia_auc_tolerance: Tolerância de AUC do MIA em torno de 0.5.
        output_dir: Diretório onde o Relatório de Prontidão é escrito.
        rng_seed: Semente aleatória, para reprodutibilidade.

    Returns:
        :class:`AuditorOutput` com os vereditos de utilidade e
        privacidade, o veredito global (privacidade tem poder de veto),
        e o caminho do Relatório de Prontidão.

    Raises:
        InsufficientHoldoutError: Se `real_holdout` for `None` ou
            demasiado pequeno — o Auditor não tem "modo sem dados reais".
    """
    if real_holdout is None or len(real_holdout) < MIN_HOLDOUT_SIZE:
        raise InsufficientHoldoutError(
            f"'real_holdout' ausente ou com menos de {MIN_HOLDOUT_SIZE} linhas — "
            "o Auditor não pode provar utilidade nem privacidade sem dados reais"
        )

    warnings: list[str] = []
    utilidade = assess_utility(
        dataset_sintetico, real_holdout, contract, target_columns, utility_threshold=utility_threshold, rng_seed=rng_seed
    )
    privacidade = assess_privacy(
        dataset_sintetico, real_holdout, contract, training_reference, dcr_ratio_threshold, mia_auc_tolerance, rng_seed
    )

    if training_reference is None:
        warnings.append(
            "'training_reference' não fornecido — o Ataque de Inferência de "
            "Pertença não foi avaliado (ver privacidade.mia.motivo_nao_avaliavel)"
        )

    aprovado_global = utilidade.aprovado and privacidade.aprovado
    diagnostico = None
    if not aprovado_global:
        motivos = []
        if not utilidade.aprovado:
            motivos.append(f"utilidade insuficiente (retenção média {utilidade.retencao_media:.1%})")
        if not privacidade.aprovado:
            motivos.append(f"privacidade reprovada ({privacidade.motivo})")
        diagnostico = "Prontidão BLOQUEADA: " + "; ".join(motivos)

    report_path = generate_readiness_report(
        utilidade, privacidade, aprovado_global, Path(output_dir) / "relatorio_prontidao.pdf"
    )

    logger.info(
        "Auditoria concluída: utilidade_aprovada=%s, privacidade_aprovada=%s, global=%s",
        utilidade.aprovado,
        privacidade.aprovado,
        aprovado_global,
    )

    return AuditorOutput(
        utilidade=utilidade,
        privacidade=privacidade,
        aprovado_global=aprovado_global,
        diagnostico=diagnostico,
        report_path=str(report_path),
        warnings=warnings,
    )
