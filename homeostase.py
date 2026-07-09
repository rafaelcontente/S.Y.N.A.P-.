"""Controlador de Homeostasia e Qualidade Distribucional (Diversidade + Contrato).

Módulo 6 do S.Y.N.A.P. Monitoriza a saúde distribucional do dataset
gerado: mantém uma média móvel exponencial (EMA) da média/variância de
cada coluna numérica, da correlação de cada par do Contrato, e das
proporções de cada coluna categórica — recalculadas a cada checkpoint
a partir do lote mais recente. A partir destas estatísticas suavizadas
calcula a divergência de Kullback-Leibler face à distribuição-alvo do
Contrato e a entropia de Shannon das colunas categóricas. Quando a
divergência suavizada excede os limiares configurados, produz
multiplicadores de peso que reforçam a seleção de fontes em regiões
sub-representadas do espaço de dados — a homeostasia do Módulo 3.

Nota de arquitectura: as estatísticas são deliberadamente uma EMA sobre
os lotes mais recentes, não uma média cumulativa desde a primeira
linha. Um sistema de homeostasia que precisa de corrigir uma deriva
"em menos de 50.000 linhas" tem de reagir ao estado *recente* do
gerador — uma média cumulativa sobre um histórico de centenas de
milhares de linhas seria dominada pelo passado e nunca convergiria no
horizonte exigido, por melhor que fosse a correção aplicada daí em
diante.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from synap.neocortex.models import CategoricalStatistics, ContinuousStatistics, StatisticalContract

from .exceptions import HomeostaseValidationError
from .models import (
    ColumnDriftReport,
    CorrelationDriftReport,
    EntropyReport,
    HealthStatus,
    HomeostasisOutput,
    HomeostasisReport,
    ReweightingAction,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

DRIFT_ALPHA_WARN: float = 0.05
"""Limiar de divergência (normalizada, suavizada) acima do qual o
sistema atua (reponderação de fontes) — o "5%" do enunciado."""

DRIFT_ALPHA_CRITICAL: float = 0.10
"""Limiar acima do qual a geração deve pausar — o "10%" do enunciado."""

ENTROPY_THRESHOLD: float = 0.7
"""Entropia de Shannon normalizada mínima para uma coluna categórica
antes de se considerar diversidade insuficiente."""

EMA_ALPHA: float = 0.3
"""Peso do lote mais recente na média móvel exponencial das
estatísticas (média, variância, correlação, proporções categóricas) —
o "filtro de média móvel suavizada" do Pilar 1/Teste 2. Valores mais
baixos suavizam mais (mais resistência a ruído amostral, mas
convergência mais lenta)."""

REINFORCEMENT_FACTOR: float = 0.5
"""Fator de reforço por omissão aplicado às fontes que corrigem uma
sub-representação — o "+50%" do exemplo do enunciado."""

SUBREPRESENTATION_RATIO: float = 0.5
"""Uma categoria é considerada sub-representada se a sua frequência
(suavizada) for inferior a esta fração da frequência-alvo do Contrato."""

MIN_STD_FLOOR: float = 1e-6
"""Piso do desvio padrão (empírico ou alvo), evita divisões/logs degenerados."""


# --------------------------------------------------------------------------
# Divergência de Kullback-Leibler (colunas numéricas, aproximação Gaussiana)
# --------------------------------------------------------------------------


def gaussian_kl(mu_empirico: float, sigma_empirico: float, mu_alvo: float, sigma_alvo: float) -> float:
    """Divergência de KL entre duas Gaussianas: `KL(empírica || alvo)`.

    Fórmula fechada: `ln(σ_alvo/σ_emp) + (σ_emp² + (μ_emp-μ_alvo)²) / (2σ_alvo²) - 0.5`.

    Nota de arquitectura: colunas declaradas como "uniforme" no Contrato
    também usam esta aproximação Gaussiana como proxy de divergência —
    o essencial do desvio prático (deslocamento de média/escala) é
    capturado sem necessitar de manter histogramas por coluna;
    simplificação documentada e deliberada.
    """
    sigma_empirico = max(sigma_empirico, MIN_STD_FLOOR)
    sigma_alvo = max(sigma_alvo, MIN_STD_FLOOR)
    return float(
        np.log(sigma_alvo / sigma_empirico)
        + (sigma_empirico**2 + (mu_empirico - mu_alvo) ** 2) / (2 * sigma_alvo**2)
        - 0.5
    )


def normalized_divergence_score(raw_kl: float) -> float:
    """Transforma uma divergência de KL bruta (nats, [0, ∞)) para [0, 1).

    Usa `1 - exp(-KL)`, monótona crescente, que permite interpretar os
    limiares de 5%/10% do enunciado como uma escala percentual limitada.
    """
    return float(1 - np.exp(-max(raw_kl, 0.0)))


# --------------------------------------------------------------------------
# Entropia de Shannon (colunas categóricas)
# --------------------------------------------------------------------------


def shannon_entropy_normalized(proportions: dict[str, float]) -> float:
    """Entropia de Shannon normalizada (0 a 1) de uma distribuição de proporções."""
    n_categorias = len(proportions)
    if n_categorias <= 1:
        return 1.0
    probs = np.array([p for p in proportions.values() if p > 0])
    if probs.sum() <= 0:
        return 1.0
    probs = probs / probs.sum()
    entropia = float(-np.sum(probs * np.log2(probs)))
    return entropia / np.log2(n_categorias)


def find_underrepresented_categories(
    proportions: dict[str, float], target_freq: dict[str, float], ratio: float = SUBREPRESENTATION_RATIO
) -> list[str]:
    """Identifica categorias cuja proporção (suavizada) é `< ratio * frequência-alvo`."""
    return [
        categoria
        for categoria, freq_alvo in target_freq.items()
        if proportions.get(categoria, 0.0) < ratio * freq_alvo
    ]


# --------------------------------------------------------------------------
# Estado do controlador (entre checkpoints sucessivos)
# --------------------------------------------------------------------------


class HomeostaseState:
    """Estado acumulado do Módulo 6 entre checkpoints sucessivos.

    Mantido pelo chamador e passado de volta a cada chamada de
    :func:`run_controlador_homeostasia`. Mantém uma EMA da média e
    variância de cada coluna numérica, da correlação de cada par
    configurado no Contrato, e das proporções de cada coluna
    categórica — tudo recalculado a partir do lote mais recente em
    cada `update()`, nunca de um histórico bruto acumulado.
    """

    def __init__(self, contract: StatisticalContract, ema_alpha: float = EMA_ALPHA) -> None:
        self.ema_alpha = ema_alpha
        self.numeric_cols = [
            c for c, s in contract.column_stats.items() if isinstance(s, ContinuousStatistics)
        ]
        self.categorical_cols = [
            c for c, s in contract.column_stats.items() if isinstance(s, CategoricalStatistics)
        ]
        self.correlation_pairs = [
            (par.coluna_a, par.coluna_b)
            for par in contract.correlations
            if par.coluna_a in self.numeric_cols and par.coluna_b in self.numeric_cols
        ]

        self.mean_ema: dict[str, float] = {}
        self.var_ema: dict[str, float] = {}
        self.corr_ema: dict[tuple[str, str], float] = {}
        self.prop_ema: dict[str, dict[str, float]] = {
            c: dict.fromkeys(contract.column_stats[c].frequencias, 0.0) for c in self.categorical_cols
        }
        self.last_batch_mean: dict[str, float] = {}
        self.last_batch_std: dict[str, float] = {}
        self.n_total: int = 0

    def update(self, df: pd.DataFrame) -> None:
        """Incorpora o lote mais recente, avançando a EMA de cada estatística."""
        for coluna in self.numeric_cols:
            if coluna not in df.columns:
                continue
            batch_mean = float(df[coluna].mean())
            batch_var = float(df[coluna].var(ddof=0)) if len(df) > 1 else 0.0
            self.last_batch_mean[coluna] = batch_mean
            self.last_batch_std[coluna] = float(np.sqrt(max(batch_var, 0.0)))

            if coluna not in self.mean_ema:
                self.mean_ema[coluna] = batch_mean
                self.var_ema[coluna] = batch_var
            else:
                self.mean_ema[coluna] = self.ema_alpha * batch_mean + (1 - self.ema_alpha) * self.mean_ema[coluna]
                self.var_ema[coluna] = self.ema_alpha * batch_var + (1 - self.ema_alpha) * self.var_ema[coluna]

        for a, b in self.correlation_pairs:
            if a not in df.columns or b not in df.columns or len(df) < 2:
                continue
            batch_corr = df[[a, b]].corr(method="pearson").loc[a, b]
            if pd.isna(batch_corr):
                continue
            chave = (a, b)
            if chave not in self.corr_ema:
                self.corr_ema[chave] = float(batch_corr)
            else:
                self.corr_ema[chave] = self.ema_alpha * float(batch_corr) + (1 - self.ema_alpha) * self.corr_ema[chave]

        for coluna in self.categorical_cols:
            if coluna not in df.columns:
                continue
            proporcoes_lote = df[coluna].value_counts(normalize=True).to_dict()
            atual = self.prop_ema[coluna]
            ja_inicializado = self.n_total > 0
            for categoria in atual:
                p_lote = proporcoes_lote.get(categoria, 0.0)
                atual[categoria] = (
                    self.ema_alpha * p_lote + (1 - self.ema_alpha) * atual[categoria]
                    if ja_inicializado
                    else p_lote
                )

        self.n_total += len(df)


# --------------------------------------------------------------------------
# Verificação de homeostasia
# --------------------------------------------------------------------------


def run_homeostasis_check(
    state: HomeostaseState,
    contract: StatisticalContract,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    drift_warn: float = DRIFT_ALPHA_WARN,
    drift_critical: float = DRIFT_ALPHA_CRITICAL,
) -> HomeostasisReport:
    """Calcula o relatório de saúde distribucional a partir do estado acumulado."""
    divergencias_colunas: list[ColumnDriftReport] = []
    divergencias_correlacao: list[CorrelationDriftReport] = []
    entropias: list[EntropyReport] = []
    maximos_suavizados = [0.0]

    for coluna in state.numeric_cols:
        if coluna not in state.mean_ema:
            continue
        stat_alvo = contract.column_stats[coluna]
        mu_suavizado = state.mean_ema[coluna]
        sigma_suavizado = float(np.sqrt(max(state.var_ema[coluna], 0.0)))

        raw_suavizado = gaussian_kl(mu_suavizado, sigma_suavizado, stat_alvo.media, stat_alvo.desvio)
        normalizada_suavizada = normalized_divergence_score(raw_suavizado)

        # divergência "instantânea": usa só o último lote, sem EMA — serve
        # para comparação/reporte (ver Pilar 1/Teste 2), a decisão usa
        # sempre a versão suavizada.
        raw_instantaneo = gaussian_kl(
            state.last_batch_mean[coluna], state.last_batch_std[coluna], stat_alvo.media, stat_alvo.desvio
        )
        normalizada_instantanea = normalized_divergence_score(raw_instantaneo)

        estado = (
            HealthStatus.CRITICO
            if normalizada_suavizada > drift_critical
            else HealthStatus.ATENCAO
            if normalizada_suavizada > drift_warn
            else HealthStatus.NORMAL
        )
        divergencias_colunas.append(
            ColumnDriftReport(
                coluna=coluna,
                divergencia_bruta=raw_suavizado,
                divergencia_normalizada=normalizada_instantanea,
                divergencia_suavizada=normalizada_suavizada,
                estado=estado,
            )
        )
        maximos_suavizados.append(normalizada_suavizada)

    for par in contract.correlations:
        chave = (par.coluna_a, par.coluna_b)
        if chave not in state.corr_ema:
            continue
        emp = state.corr_ema[chave]
        desvio_abs = abs(emp - par.valor)
        divergencias_correlacao.append(
            CorrelationDriftReport(
                coluna_a=par.coluna_a,
                coluna_b=par.coluna_b,
                correlacao_alvo=par.valor,
                correlacao_observada=emp,
                desvio_absoluto=desvio_abs,
            )
        )
        maximos_suavizados.append(min(desvio_abs, 1.0))

    for coluna in state.categorical_cols:
        proporcoes = state.prop_ema[coluna]
        freq_alvo = contract.column_stats[coluna].frequencias
        entropia = shannon_entropy_normalized(proporcoes)
        sub_representadas = find_underrepresented_categories(proporcoes, freq_alvo)
        entropias.append(
            EntropyReport(
                coluna=coluna,
                entropia_normalizada=entropia,
                abaixo_do_limiar=entropia < entropy_threshold,
                categorias_sub_representadas=sub_representadas,
            )
        )

    divergencia_global = max(maximos_suavizados)
    estado_global = (
        HealthStatus.CRITICO
        if divergencia_global > drift_critical
        else HealthStatus.ATENCAO
        if divergencia_global > drift_warn
        else HealthStatus.NORMAL
    )

    return HomeostasisReport(
        n_linhas_acumuladas=state.n_total,
        divergencias_colunas=divergencias_colunas,
        divergencias_correlacao=divergencias_correlacao,
        entropias=entropias,
        divergencia_global_suavizada=divergencia_global,
        estado_global=estado_global,
    )


# --------------------------------------------------------------------------
# Reponderação das fontes do Módulo 3
# --------------------------------------------------------------------------


def compute_reweighting(
    memory: pd.DataFrame,
    contract: StatisticalContract,
    report: HomeostasisReport,
    state: HomeostaseState,
    reinforcement_factor: float = REINFORCEMENT_FACTOR,
) -> tuple[np.ndarray, list[ReweightingAction]]:
    """Calcula multiplicadores de peso que corrigem sub-representações na memória.

    Para cada coluna numérica em deriva (estado != NORMAL), reforça as
    linhas da memória por razão de verosimilhança (Gaussiana-alvo sobre
    Gaussiana suavizada atual) — corrige média E variância em conjunto,
    ao contrário de simplesmente empurrar para a cauda extrema. Para
    cada coluna categórica com diversidade insuficiente, reforça as
    linhas que contêm categorias sub-representadas.

    Returns:
        Tuplo (multiplicadores_por_linha, acoes_aplicadas).
    """
    multiplicador = np.ones(len(memory))
    acoes: list[ReweightingAction] = []

    for col_report in report.divergencias_colunas:
        if col_report.estado == HealthStatus.NORMAL:
            continue
        coluna = col_report.coluna
        if coluna not in memory.columns or coluna not in state.mean_ema:
            continue
        target_media = contract.column_stats[coluna].media
        target_desvio = max(contract.column_stats[coluna].desvio, MIN_STD_FLOOR)
        emp_media = state.mean_ema[coluna]
        emp_desvio = max(float(np.sqrt(max(state.var_ema[coluna], 0.0))), MIN_STD_FLOOR)

        x = memory[coluna].to_numpy(dtype=float)
        log_alvo = -0.5 * ((x - target_media) / target_desvio) ** 2 - np.log(target_desvio)
        log_emp = -0.5 * ((x - emp_media) / emp_desvio) ** 2 - np.log(emp_desvio)
        log_razao = log_alvo - log_emp
        razao = np.exp(log_razao - log_razao.max())
        razao = razao / max(razao.mean(), MIN_STD_FLOOR)

        fator = 1.0 + reinforcement_factor * (razao - 1.0)
        multiplicador *= np.clip(fator, 0.01, None)
        acoes.append(
            ReweightingAction(
                coluna_alvo=coluna, motivo="deriva_distribucional", fator_reforco=reinforcement_factor
            )
        )

    for ent_report in report.entropias:
        if not ent_report.categorias_sub_representadas:
            continue
        coluna = ent_report.coluna
        if coluna not in memory.columns:
            continue
        mask = memory[coluna].isin(ent_report.categorias_sub_representadas).to_numpy()
        multiplicador *= np.where(mask, 1.0 + reinforcement_factor, 1.0)
        acoes.append(
            ReweightingAction(coluna_alvo=coluna, motivo="entropia_baixa", fator_reforco=reinforcement_factor)
        )

    return multiplicador, acoes


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def run_controlador_homeostasia(
    new_batch: pd.DataFrame,
    contract: StatisticalContract,
    memory: pd.DataFrame,
    state: HomeostaseState,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    drift_warn: float = DRIFT_ALPHA_WARN,
    drift_critical: float = DRIFT_ALPHA_CRITICAL,
    reinforcement_factor: float = REINFORCEMENT_FACTOR,
) -> HomeostasisOutput:
    """Processa um checkpoint de homeostasia (Módulo 6 do S.Y.N.A.P.).

    Args:
        new_batch: Linhas aprovadas desde o último checkpoint (ex.: a
            saída do Módulo 5).
        contract: O Contrato Estatístico do Módulo 1.
        memory: A memória vetorial atual (Módulo 3/5), usada para
            calcular os multiplicadores de reponderação.
        state: Estado acumulado entre checkpoints (ver :class:`HomeostaseState`).
        entropy_threshold: Entropia normalizada mínima aceite.
        drift_warn: Limiar de divergência suavizada para reponderação.
        drift_critical: Limiar de divergência suavizada para recomendar pausa.
        reinforcement_factor: Fator de reforço das ações de reponderação.

    Returns:
        :class:`HomeostasisOutput` com o relatório, as ações e os
        multiplicadores de peso deste checkpoint.

    Raises:
        HomeostaseValidationError: Se `new_batch` estiver vazio.
    """
    if len(new_batch) == 0:
        raise HomeostaseValidationError("new_batch está vazio — nada para atualizar a homeostasia")

    state.update(new_batch)
    relatorio = run_homeostasis_check(state, contract, entropy_threshold, drift_warn, drift_critical)
    multiplicador, acoes = compute_reweighting(memory, contract, relatorio, state, reinforcement_factor)

    warnings: list[str] = []
    pausa_recomendada = relatorio.estado_global == HealthStatus.CRITICO
    if pausa_recomendada:
        warnings.append(
            f"DIVERGÊNCIA CRÍTICA ({relatorio.divergencia_global_suavizada:.1%} > "
            f"{drift_critical:.0%}) após {relatorio.n_linhas_acumuladas} linhas — "
            "recomenda-se pausar a geração e solicitar nova semente ou "
            "relaxamento do Contrato"
        )
    elif relatorio.estado_global == HealthStatus.ATENCAO:
        colunas_afetadas = [c.coluna for c in relatorio.divergencias_colunas if c.estado != HealthStatus.NORMAL]
        warnings.append(
            f"deriva distribucional detectada em {colunas_afetadas} — "
            "pesos de seleção de fontes reforçados"
        )
    for entropia in relatorio.entropias:
        if entropia.abaixo_do_limiar:
            warnings.append(
                f"diversidade insuficiente em '{entropia.coluna}' "
                f"(entropia={entropia.entropia_normalizada:.2f} < {entropy_threshold}) — "
                "reforçando fontes com categorias raras"
            )

    logger.info(
        "Homeostasia verificada: n=%d, estado_global=%s, divergencia=%.4f, acoes=%d",
        relatorio.n_linhas_acumuladas,
        relatorio.estado_global.value,
        relatorio.divergencia_global_suavizada,
        len(acoes),
    )

    return HomeostasisOutput(
        relatorio=relatorio,
        acoes=acoes,
        pesos_multiplicador=list(multiplicador),
        pausa_recomendada=pausa_recomendada,
        warnings=warnings,
    )
