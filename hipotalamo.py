"""Hipotálamo de Raciocínio Lógico e Validador Causal (ASP + ILP + Testes de Plausibilidade).

Módulo 4 do S.Y.N.A.P. Opera em três frentes sobre cada lote recebido
do Módulo 3: (1) validação estrita das regras de negócio, com traço de
prova exato por violação; (2) validação causal contínua, recalculando
a correlação empírica acumulada a cada `checkpoint_size` linhas e
testando-a contra o Contrato e a DAG para detetar deriva; (3) indução
de novas regras (ILP) suportadas por evidência estatística e
logicamente consistentes (zero exceções), consolidadas como
conhecimento permanente apenas quando não duplicam a estrutura causal
já conhecida.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from synap.hipocampo.hipocampo import fisher_z_p_value, partial_correlation
from synap.hipocampo.models import CausalDAG
from synap.neocortex.models import (
    BusinessRule,
    CategoricalStatistics,
    ContinuousStatistics,
    RuleOperator,
    StatisticalContract,
)

from .exceptions import HipotalamoValidationError
from .models import (
    BatchValidationReport,
    CandidateRule,
    CausalDriftAlert,
    CausalMonitoringReport,
    ConsolidatedRule,
    HipotalamoOutput,
    ILPReport,
    ProofTrace,
    RowValidationResult,
    RuleValidationOutcome,
    ValidationVerdict,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

DEFAULT_CHECKPOINT_SIZE: int = 5_000
"""Nº de linhas aprovadas acumuladas entre verificações de deriva causal e ILP."""

DEFAULT_DRIFT_ALPHA: float = 0.01
"""Nível de significância dos testes de deriva causal."""

MIN_SUPPORT_ILP: int = 100
"""Suporte mínimo (nº de exemplos) para uma regra candidata ser considerada."""

MIN_CONFIDENCE_ILP: float = 0.98
"""Confiança mínima (P(consequente | antecedente)) para propor uma candidata."""

ILP_CANDIDATE_QUANTILES: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
"""Quantis testados como limiar candidato na condição numérica do ILP.

O motor de indução está deliberadamente restrito ao template
`SE coluna_numerica operador limiar ENTÃO coluna_categorica = valor`
— o padrão do exemplo do enunciado — e não a um ILP genérico
multivariado, para manter o espaço de busca e o custo computacional
controlados por checkpoint."""

ILP_BUFFER_MAX_SIZE: int = 20_000
"""Teto do buffer de linhas aprovadas retido para indução de padrões (ILP).

A validação causal contínua usa estatísticas suficientes acumuladas
(ver :class:`RunningCovarianceAccumulator`, custo O(1) por linha), mas
a indução de regras precisa de valores brutos — por isso mantém-se
apenas uma amostra-reservatório limitada, não o histórico completo."""

_OPERATOR_FUNCS = {
    RuleOperator.GT: lambda s, v: s > v,
    RuleOperator.LT: lambda s, v: s < v,
    RuleOperator.GE: lambda s, v: s >= v,
    RuleOperator.LE: lambda s, v: s <= v,
    RuleOperator.EQ: lambda s, v: s == v,
    RuleOperator.NE: lambda s, v: s != v,
}


# --------------------------------------------------------------------------
# Frente 1 — Validação Estrita (ASP)
# --------------------------------------------------------------------------


def validate_row(
    row: dict[str, Any], rules: list[BusinessRule]
) -> list[ProofTrace]:
    """Valida uma única linha contra todas as regras de negócio.

    Args:
        row: A linha a validar, como mapa coluna -> valor.
        rules: As regras de negócio (do Contrato Estatístico) a impor.

    Returns:
        Lista de :class:`ProofTrace`, uma por regra violada (vazia se
        a linha respeita todas as regras).
    """
    provas: list[ProofTrace] = []
    for regra in rules:
        if regra.coluna not in row:
            continue
        valor = row[regra.coluna]
        if not _OPERATOR_FUNCS[regra.operador](valor, regra.valor):
            provas.append(
                ProofTrace(
                    regra_violada=regra.texto_original,
                    clausula=(
                        f"{regra.coluna}(L,{valor!r}) → contradiz "
                        f"'{regra.texto_original}'"
                    ),
                    valor_observado=valor,
                )
            )
    return provas


def validate_batch(
    df: pd.DataFrame, rules: list[BusinessRule]
) -> tuple[BatchValidationReport, list[RowValidationResult]]:
    """Valida todas as linhas de um lote contra as regras de negócio.

    Implementação vetorizada: cada regra é avaliada de uma só vez sobre
    toda a coluna (não linha a linha), e o traço de prova só é
    construído para as linhas efetivamente violadoras — essencial para
    escalar a lotes de milhões de linhas sem percorrer cada uma
    individualmente quando (como é o caso típico) a esmagadora maioria
    respeita as regras.

    Args:
        df: O lote a validar (ex.: a saída do Módulo 3).
        rules: As regras de negócio a impor.

    Returns:
        Tuplo (relatorio_agregado, resultados_das_linhas_rejeitadas).
        Linhas aprovadas não geram registo individual (ver
        :class:`RowValidationResult`).
    """
    total = len(df)
    if not rules:
        return (
            BatchValidationReport(
                total_linhas=total, aprovadas=total, rejeitadas=0, taxa_rejeicao=0.0
            ),
            [],
        )

    provas_por_linha: dict[Any, list[ProofTrace]] = {}
    for regra in rules:
        if regra.coluna not in df.columns:
            continue
        coluna = df[regra.coluna]
        valido = _OPERATOR_FUNCS[regra.operador](coluna, regra.valor)
        for indice in df.index[~valido]:
            valor = coluna.loc[indice]
            provas_por_linha.setdefault(indice, []).append(
                ProofTrace(
                    regra_violada=regra.texto_original,
                    clausula=(
                        f"{regra.coluna}(L,{valor!r}) → contradiz "
                        f"'{regra.texto_original}'"
                    ),
                    valor_observado=valor,
                )
            )

    rejeicoes = [
        RowValidationResult(indice=int(indice), veredito=ValidationVerdict.REJEITADO, provas=provas)
        for indice, provas in provas_por_linha.items()
    ]
    aprovadas = total - len(rejeicoes)
    relatorio = BatchValidationReport(
        total_linhas=total,
        aprovadas=aprovadas,
        rejeitadas=len(rejeicoes),
        taxa_rejeicao=(len(rejeicoes) / total) if total else 0.0,
    )
    return relatorio, rejeicoes


# --------------------------------------------------------------------------
# Estatísticas suficientes acumuladas (para a Frente 2, custo O(1)/linha)
# --------------------------------------------------------------------------


class RunningCovarianceAccumulator:
    """Acumula somas suficientes para recalcular médias/correlações incrementalmente.

    Mantém `n`, a soma e a soma de produtos cruzados por par de colunas
    numéricas, permitindo recalcular a matriz de correlação a qualquer
    momento em O(p²) sem guardar as linhas brutas — essencial para
    monitorizar a deriva causal ao longo de milhões de linhas geradas
    sem esgotar RAM.

    Nota de arquitectura: este é o único objeto deliberadamente
    mutável do módulo (estado incremental por natureza). O método
    `soma_de_quadrados_e_produtos_cruzados` é numericamente menos
    estável do que o algoritmo de Welford para `n` muito grande ou
    valores de grande magnitude; para os volumes tratados aqui é
    suficiente, e a troca é documentada deliberadamente.
    """

    def __init__(self, numeric_columns: list[str]) -> None:
        self.numeric_columns = numeric_columns
        self.n: int = 0
        self._soma: dict[str, float] = {c: 0.0 for c in numeric_columns}
        self._soma_produtos: dict[tuple[str, str], float] = {
            (a, b): 0.0 for a in numeric_columns for b in numeric_columns
        }

    def update(self, df: pd.DataFrame) -> None:
        """Incorpora um novo lote de linhas nas estatísticas acumuladas."""
        if not self.numeric_columns:
            self.n += len(df)
            return
        dados = df[self.numeric_columns].to_numpy(dtype=float)
        self.n += len(df)
        for i, col_a in enumerate(self.numeric_columns):
            self._soma[col_a] += float(dados[:, i].sum())
            for j, col_b in enumerate(self.numeric_columns):
                if j < i:
                    continue
                produto = float((dados[:, i] * dados[:, j]).sum())
                self._soma_produtos[(col_a, col_b)] += produto
                self._soma_produtos[(col_b, col_a)] += produto if col_a != col_b else 0.0

    def mean(self, col: str) -> float:
        return self._soma[col] / self.n if self.n else 0.0

    def covariance_matrix(self) -> pd.DataFrame:
        """Devolve a matriz de covariância acumulada."""
        cols = self.numeric_columns
        matriz = pd.DataFrame(0.0, index=cols, columns=cols)
        for a in cols:
            for b in cols:
                cov = self._soma_produtos[(a, b)] / self.n - self.mean(a) * self.mean(b)
                matriz.loc[a, b] = cov
        return matriz

    def correlation_matrix(self) -> pd.DataFrame:
        """Devolve a matriz de correlação acumulada, derivada da covariância."""
        cov = self.covariance_matrix()
        cols = self.numeric_columns
        matriz = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        for a in cols:
            for b in cols:
                if a == b:
                    continue
                denom = np.sqrt(max(cov.loc[a, a], 0.0) * max(cov.loc[b, b], 0.0))
                matriz.loc[a, b] = cov.loc[a, b] / denom if denom > 0 else 0.0
        return matriz


# --------------------------------------------------------------------------
# Frente 2 — Validação Causal Contínua
# --------------------------------------------------------------------------


def fisher_z_comparison_p_value(r1: float, n1: int, r2: float, n2: int) -> float:
    """Testa H0: as correlações `r1` e `r2` (de amostras de tamanho `n1`, `n2`) são iguais.

    Usa o teste de comparação de duas correlações de Fisher-Z, base da
    deteção de "correlação quebrada" (deriva face ao Contrato original).

    Returns:
        P-valor bicaudal. Devolve 1.0 se os graus de liberdade forem insuficientes.
    """
    if n1 <= 3 or n2 <= 3:
        return 1.0
    r1_seguro = float(np.clip(r1, -0.999999, 0.999999))
    r2_seguro = float(np.clip(r2, -0.999999, 0.999999))
    z1 = np.arctanh(r1_seguro)
    z2 = np.arctanh(r2_seguro)
    erro_padrao = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / erro_padrao
    return float(2 * (1 - scipy_stats.norm.cdf(abs(z))))


def run_causal_monitoring(
    contract: StatisticalContract,
    dag: CausalDAG,
    empirical_correlation: pd.DataFrame,
    n_accumulated: int,
    alpha: float = DEFAULT_DRIFT_ALPHA,
) -> CausalMonitoringReport:
    """Verifica se a DAG e as correlações do Contrato continuam válidas.

    Executa dois tipos de teste:

    1. **Correlação desviada**: para cada par com correlação-alvo no
       Contrato, compara-a estatisticamente (Fisher-Z) com a
       correlação empírica acumulada — deteta arestas que enfraqueceram
       ou inverteram face ao esperado.
    2. **Dependência emergente**: para cada par NÃO-adjacente na DAG,
       testa a correlação parcial condicionada nos pais de ambos (como
       no Módulo 2) — deteta relações novas que a DAG não previa.

    Args:
        contract: O Contrato Estatístico original (fonte da verdade estatística).
        dag: A DAG atual (do Módulo 2, possivelmente já ajustada).
        empirical_correlation: Matriz de correlação recalculada sobre
            os dados acumulados gerados (ver :class:`RunningCovarianceAccumulator`).
        n_accumulated: Tamanho da amostra acumulada.
        alpha: Nível de significância dos testes.

    Returns:
        :class:`CausalMonitoringReport` com todos os alertas de deriva detectados.
    """
    alertas: list[CausalDriftAlert] = []
    verificacoes = 0
    numeric_cols = list(empirical_correlation.columns)

    # --- Teste 1: correlação desviada face ao Contrato ---
    for par in contract.correlations:
        if par.coluna_a not in numeric_cols or par.coluna_b not in numeric_cols:
            continue
        r_empirico = empirical_correlation.loc[par.coluna_a, par.coluna_b]
        if pd.isna(r_empirico):
            continue
        verificacoes += 1
        p_valor = fisher_z_comparison_p_value(
            par.valor, contract.sample_size or n_accumulated, float(r_empirico), n_accumulated
        )
        if p_valor < alpha:
            alertas.append(
                CausalDriftAlert(
                    coluna_a=par.coluna_a,
                    coluna_b=par.coluna_b,
                    tipo="correlacao_desviada",
                    correlacao_observada=float(r_empirico),
                    p_valor=p_valor,
                    n_linhas_acumuladas=n_accumulated,
                )
            )

    # --- Teste 2: dependência emergente (pares não-adjacentes na DAG) ---
    for i, x in enumerate(numeric_cols):
        for y in numeric_cols[i + 1 :]:
            if dag.is_adjacent(x, y):
                continue
            conditioning_set = sorted(
                (set(dag.parents(x)) | set(dag.parents(y)) - {x, y}) & set(numeric_cols)
            )
            verificacoes += 1
            rho = partial_correlation(empirical_correlation, x, y, conditioning_set)
            p_valor = fisher_z_p_value(rho, n_accumulated, len(conditioning_set))
            if p_valor < alpha:
                alertas.append(
                    CausalDriftAlert(
                        coluna_a=x,
                        coluna_b=y,
                        tipo="dependencia_emergente",
                        correlacao_observada=rho,
                        p_valor=p_valor,
                        n_linhas_acumuladas=n_accumulated,
                    )
                )

    return CausalMonitoringReport(
        verificacoes_realizadas=verificacoes,
        derivas_detectadas=alertas,
        n_linhas_analisadas=n_accumulated,
    )


# --------------------------------------------------------------------------
# Frente 3 — Indução de Regras (ILP) com Regularização
# --------------------------------------------------------------------------


def induce_rules(
    buffer: pd.DataFrame,
    contract: StatisticalContract,
    dag: CausalDAG,
    min_support: int = MIN_SUPPORT_ILP,
    min_confidence: float = MIN_CONFIDENCE_ILP,
) -> tuple[list[CandidateRule], int]:
    """Induz regras candidatas do padrão `SE numerica > limiar ENTÃO categorica = valor`.

    Descarta automaticamente candidatas cuja relação já é um subproduto
    direto da DAG (aresta `coluna_condicao -> coluna_alvo` já existente)
    — evita duplicar conhecimento causal já representado estruturalmente.

    Args:
        buffer: Amostra de linhas aprovadas retida para indução (ver `ILP_BUFFER_MAX_SIZE`).
        contract: O Contrato Estatístico (identifica colunas numéricas/categóricas).
        dag: A DAG atual, usada para filtrar subprodutos diretos.
        min_support: Suporte mínimo (nº de exemplos) para considerar uma candidata.
        min_confidence: Confiança mínima para propor uma candidata.

    Returns:
        Tuplo (candidatas_novas, numero_descartado_por_ja_estar_na_dag).
    """
    numeric_cols = [
        c for c in buffer.columns if isinstance(contract.column_stats[c], ContinuousStatistics)
    ]
    categorical_cols = [
        c for c in buffer.columns if isinstance(contract.column_stats[c], CategoricalStatistics)
    ]

    candidatas: list[CandidateRule] = []
    descartadas_por_dag = 0

    for condicao in numeric_cols:
        limiares = buffer[condicao].quantile(list(ILP_CANDIDATE_QUANTILES)).unique()
        for alvo in categorical_cols:
            if condicao == alvo:
                continue
            ja_existe = dag.is_adjacent(condicao, alvo)
            for limiar in limiares:
                mascara = buffer[condicao] > limiar
                suporte = int(mascara.sum())
                if suporte < min_support:
                    continue
                for valor_alvo in contract.column_stats[alvo].frequencias:
                    confianca = float((buffer.loc[mascara, alvo] == valor_alvo).mean())
                    if confianca < min_confidence:
                        continue
                    if ja_existe:
                        descartadas_por_dag += 1
                        continue
                    candidatas.append(
                        CandidateRule(
                            coluna_condicao=condicao,
                            operador=RuleOperator.GT,
                            limiar=float(limiar),
                            coluna_alvo=alvo,
                            valor_alvo=valor_alvo,
                            suporte=suporte,
                            confianca=confianca,
                        )
                    )

    return candidatas, descartadas_por_dag


def validate_candidate_with_asp(
    candidate: CandidateRule, data: pd.DataFrame
) -> RuleValidationOutcome:
    """Valida uma regra candidata como restrição lógica rígida (zero exceções).

    A candidata só é consolidada como conhecimento permanente se, ao
    ser imposta como regra rígida sobre `data`, não existir NENHUMA
    exceção — a "confiança estatística" do ILP (que tolera pequenas
    exceções, `min_confidence < 1.0`) é assim distinguida da
    consistência lógica exigida para uma regra permanente do ASP.

    Returns:
        :class:`RuleValidationOutcome` com o veredito e o número de exceções.
    """
    antecedente = _OPERATOR_FUNCS[candidate.operador](data[candidate.coluna_condicao], candidate.limiar)
    consequente = data[candidate.coluna_alvo] == candidate.valor_alvo
    violacoes = int((antecedente & ~consequente).sum())

    if violacoes == 0:
        return RuleValidationOutcome(candidata=candidate, consolidada=True, violacoes=0)
    return RuleValidationOutcome(
        candidata=candidate,
        consolidada=False,
        violacoes=violacoes,
        motivo_rejeicao=(
            f"{violacoes} exceção(ões) encontrada(s) na validação ASP "
            "(regras permanentes exigem consistência lógica total)"
        ),
    )


def run_ilp(
    buffer: pd.DataFrame,
    contract: StatisticalContract,
    dag: CausalDAG,
    min_support: int = MIN_SUPPORT_ILP,
    min_confidence: float = MIN_CONFIDENCE_ILP,
) -> tuple[ILPReport, list[ConsolidatedRule]]:
    """Executa o ciclo completo de indução + validação ASP num checkpoint.

    Returns:
        Tuplo (relatorio_ilp, regras_consolidadas_prontas_para_o_hipocampo).
    """
    candidatas, descartadas_por_dag = induce_rules(buffer, contract, dag, min_support, min_confidence)

    consolidadas: list[RuleValidationOutcome] = []
    rejeitadas: list[RuleValidationOutcome] = []
    novas_regras: list[ConsolidatedRule] = []

    for candidata in candidatas:
        resultado = validate_candidate_with_asp(candidata, buffer)
        if resultado.consolidada:
            consolidadas.append(resultado)
            novas_regras.append(
                ConsolidatedRule(
                    coluna_condicao=candidata.coluna_condicao,
                    operador=candidata.operador,
                    limiar=candidata.limiar,
                    coluna_alvo=candidata.coluna_alvo,
                    valor_alvo=candidata.valor_alvo,
                    suporte=candidata.suporte,
                    confianca=candidata.confianca,
                )
            )
        else:
            rejeitadas.append(resultado)

    relatorio = ILPReport(
        candidatas_avaliadas=len(candidatas),
        descartadas_por_dag=descartadas_por_dag,
        consolidadas=consolidadas,
        rejeitadas=rejeitadas,
    )
    return relatorio, novas_regras


# --------------------------------------------------------------------------
# Estado do orquestrador (entre lotes sucessivos)
# --------------------------------------------------------------------------


class HipotalamoState:
    """Estado acumulado do Módulo 4 entre lotes sucessivos.

    Mantido pelo chamador (ex.: o orquestrador principal do S.Y.N.A.P.)
    e passado de volta a cada chamada de :func:`run_hipotalamo`. Agrupa
    o único estado mutável necessário: as estatísticas suficientes
    acumuladas (custo O(1)/linha) e o buffer-reservatório para ILP.
    """

    def __init__(self, contract: StatisticalContract) -> None:
        numeric_cols = [
            c for c, s in contract.column_stats.items() if isinstance(s, ContinuousStatistics)
        ]
        self.accumulator = RunningCovarianceAccumulator(numeric_cols)
        self.ilp_buffer: pd.DataFrame = pd.DataFrame(columns=list(contract.column_stats))
        self.regras_consolidadas: list[ConsolidatedRule] = []
        self.total_aprovadas: int = 0
        self.ultimo_checkpoint: int = 0

    def _append_to_ilp_buffer(self, novas_linhas: pd.DataFrame, rng: np.random.Generator) -> None:
        combinado = pd.concat([self.ilp_buffer, novas_linhas], ignore_index=True)
        if len(combinado) > ILP_BUFFER_MAX_SIZE:
            indices = rng.choice(len(combinado), size=ILP_BUFFER_MAX_SIZE, replace=False)
            combinado = combinado.iloc[np.sort(indices)].reset_index(drop=True)
        self.ilp_buffer = combinado


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def run_hipotalamo(
    new_batch: pd.DataFrame,
    contract: StatisticalContract,
    dag: CausalDAG,
    state: HipotalamoState,
    checkpoint_size: int = DEFAULT_CHECKPOINT_SIZE,
    drift_alpha: float = DEFAULT_DRIFT_ALPHA,
    min_support_ilp: int = MIN_SUPPORT_ILP,
    min_confidence_ilp: float = MIN_CONFIDENCE_ILP,
    rng_seed: int | None = None,
) -> HipotalamoOutput:
    """Processa um lote do Módulo 3 através das três frentes do Hipotálamo.

    Args:
        new_batch: O lote de linhas candidatas (ex.: a saída do Expansor).
        contract: O Contrato Estatístico do Módulo 1 (regras + correlações-alvo).
        dag: A DAG atual do Módulo 2.
        state: Estado acumulado entre lotes (ver :class:`HipotalamoState`);
            é atualizado (mutado) durante esta chamada.
        checkpoint_size: Nº de linhas aprovadas acumuladas entre
            verificações de deriva causal e ciclos de ILP.
        drift_alpha: Nível de significância da deteção de deriva causal.
        min_support_ilp: Suporte mínimo para candidatas do ILP.
        min_confidence_ilp: Confiança mínima para candidatas do ILP.
        rng_seed: Semente aleatória para a amostragem do buffer de ILP.

    Returns:
        :class:`HipotalamoOutput` com a validação, alertas e novas regras deste lote.

    Raises:
        HipotalamoValidationError: Se `new_batch` estiver vazio.
    """
    if len(new_batch) == 0:
        raise HipotalamoValidationError("new_batch está vazio — nada a validar")

    warnings: list[str] = []
    rng = np.random.default_rng(rng_seed)

    # --- Frente 1: validação estrita (ASP) ---
    relatorio_validacao, rejeicoes = validate_batch(new_batch, contract.rules)
    indices_aprovados = [i for i in new_batch.index if i not in {r.indice for r in rejeicoes}]
    aprovadas_df = new_batch.loc[indices_aprovados]

    monitorizacao: CausalMonitoringReport | None = None
    relatorio_ilp: ILPReport | None = None
    novas_regras: list[ConsolidatedRule] = []

    if len(aprovadas_df) > 0:
        state.accumulator.update(aprovadas_df)
        state._append_to_ilp_buffer(aprovadas_df, rng)
        state.total_aprovadas += len(aprovadas_df)

        checkpoints_ultrapassados = (
            state.total_aprovadas // checkpoint_size - state.ultimo_checkpoint // checkpoint_size
        )
        if checkpoints_ultrapassados > 0 and state.accumulator.n > 3:
            state.ultimo_checkpoint = state.total_aprovadas

            correlacao_empirica = state.accumulator.correlation_matrix()
            monitorizacao = run_causal_monitoring(
                contract, dag, correlacao_empirica, state.accumulator.n, drift_alpha
            )
            if monitorizacao.derivas_detectadas:
                warnings.append(
                    f"DERIVA CAUSAL DETECTADA após {state.accumulator.n} linhas "
                    f"acumuladas: {len(monitorizacao.derivas_detectadas)} par(es) "
                    "afetado(s) — Módulo 5 deve rever/corrigir"
                )

            if len(state.ilp_buffer) >= min_support_ilp:
                relatorio_ilp, novas_regras = run_ilp(
                    state.ilp_buffer, contract, dag, min_support_ilp, min_confidence_ilp
                )
                state.regras_consolidadas.extend(novas_regras)
                if novas_regras:
                    warnings.append(
                        f"{len(novas_regras)} nova(s) regra(s) consolidada(s) pelo "
                        "ILP e enviada(s) ao Hipocampo como conhecimento permanente"
                    )

    logger.info(
        "Hipotálamo processou lote: total=%d, aprovadas=%d, rejeitadas=%d, "
        "deriva_verificada=%s, regras_novas=%d",
        len(new_batch),
        relatorio_validacao.aprovadas,
        relatorio_validacao.rejeitadas,
        monitorizacao is not None,
        len(novas_regras),
    )

    return HipotalamoOutput(
        validacao=relatorio_validacao,
        rejeicoes=rejeicoes,
        monitorizacao_causal=monitorizacao,
        inducao=relatorio_ilp,
        novas_regras=novas_regras,
        warnings=warnings,
    )
