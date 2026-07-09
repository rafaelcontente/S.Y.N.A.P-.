"""Testes do módulo Hipotálamo de Raciocínio Lógico e Validador Causal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synap.hipocampo.hipocampo import build_dag
from synap.hipotalamo import (
    HipotalamoState,
    HipotalamoValidationError,
    RunningCovarianceAccumulator,
    fisher_z_comparison_p_value,
    induce_rules,
    run_causal_monitoring,
    run_hipotalamo,
    run_ilp,
    validate_batch,
    validate_candidate_with_asp,
    validate_row,
)
from synap.hipotalamo.models import CandidateRule
from synap.neocortex.models import (
    BusinessRule,
    CategoricalStatistics,
    ContinuousStatistics,
    ContractSource,
    CorrelationPair,
    DistributionType,
    RuleOperator,
    StatisticalContract,
)

# --------------------------------------------------------------------------
# Fixtures e helpers
# --------------------------------------------------------------------------


def _continuous(media: float, desvio: float) -> ContinuousStatistics:
    return ContinuousStatistics(
        tipo=DistributionType.NORMAL,
        media=media,
        desvio=desvio,
        minimo=media - 5 * desvio,
        maximo=media + 5 * desvio,
        quartis=(media - 0.674 * desvio, media, media + 0.674 * desvio),
    )


def _make_contract(
    column_stats: dict, correlations: list[CorrelationPair], rules: list[BusinessRule] | None = None, sample_size: int = 1000
) -> StatisticalContract:
    return StatisticalContract(
        column_stats=column_stats,
        correlations=correlations,
        tolerancia_kl=0.05,
        source=ContractSource.REAL_CSV,
        is_synthetic_pure=False,
        sample_size=sample_size,
        rules=rules or [],
        warnings=[],
    )


@pytest.fixture
def contrato_idade_saldo_com_regra() -> StatisticalContract:
    column_stats = {
        "idade": _continuous(45, 15),
        "saldo": _continuous(50_000, 20_000),
    }
    correlations = [CorrelationPair(coluna_a="idade", coluna_b="saldo", valor=0.9)]
    rules = [
        BusinessRule(coluna="idade", operador=RuleOperator.GE, valor=18, texto_original="idade >= 18")
    ]
    return _make_contract(column_stats, correlations, rules=rules)


@pytest.fixture
def dag_idade_saldo():
    return build_dag(["idade", "saldo"], [("idade", "saldo")])


def _gerar_dados_correlacionados(n: int, rho: float, rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    z_idade = rng.normal(0, 1, n)
    z_saldo = rho * z_idade + np.sqrt(max(1 - rho**2, 0)) * rng.normal(0, 1, n)
    idade = 45 + 15 * z_idade
    saldo = 50_000 + 20_000 * z_saldo
    return pd.DataFrame({"idade": idade, "saldo": saldo})


# --------------------------------------------------------------------------
# Frente 1 — Validação Estrita (ASP)
# --------------------------------------------------------------------------


def test_validate_row_deteta_violacao_com_prova(contrato_idade_saldo_com_regra) -> None:
    provas = validate_row({"idade": 17, "saldo": 5000}, contrato_idade_saldo_com_regra.rules)
    assert len(provas) == 1
    assert provas[0].regra_violada == "idade >= 18"
    assert "idade(L,17)" in provas[0].clausula
    assert "contradiz" in provas[0].clausula


def test_validate_row_linha_valida_sem_provas(contrato_idade_saldo_com_regra) -> None:
    provas = validate_row({"idade": 30, "saldo": 5000}, contrato_idade_saldo_com_regra.rules)
    assert provas == []


def test_validate_batch_relatorio_correto(contrato_idade_saldo_com_regra) -> None:
    df = pd.DataFrame({"idade": [17, 30, 16, 40], "saldo": [1000, 2000, 3000, 4000]})
    relatorio, rejeicoes = validate_batch(df, contrato_idade_saldo_com_regra.rules)

    assert relatorio.total_linhas == 4
    assert relatorio.aprovadas == 2
    assert relatorio.rejeitadas == 2
    assert relatorio.taxa_rejeicao == pytest.approx(0.5)
    assert {r.indice for r in rejeicoes} == {0, 2}


def test_validate_batch_sem_regras_aprova_tudo() -> None:
    df = pd.DataFrame({"idade": [17, 200]})
    relatorio, rejeicoes = validate_batch(df, [])
    assert relatorio.aprovadas == 2
    assert rejeicoes == []


# --------------------------------------------------------------------------
# Pilar 2 / Teste 3 — acuidade ASP a 100% (zero falsos negativos)
# --------------------------------------------------------------------------


def test_asp_zero_falsos_negativos_em_lote_grande_e_misto(
    contrato_idade_saldo_com_regra, dag_idade_saldo
) -> None:
    """Constrói um lote com um conjunto CONHECIDO de violações intercaladas
    com linhas válidas e corre o pipeline COMPLETO (ASP + monitorização +
    ILP); confirma que TODAS as violações conhecidas são rejeitadas,
    mesmo com os outros subsistemas ativos no mesmo lote.
    """
    n = 6000
    df = _gerar_dados_correlacionados(n, rho=0.9, rng_seed=11)
    # injeta violações conhecidas e determinísticas em índices espaçados
    indices_violadores = list(range(0, n, 7))  # ~857 violações conhecidas
    df.loc[indices_violadores, "idade"] = 15  # viola 'idade >= 18'

    estado = HipotalamoState(contrato_idade_saldo_com_regra)
    resultado = run_hipotalamo(
        df, contrato_idade_saldo_com_regra, dag_idade_saldo, estado, checkpoint_size=5000
    )

    indices_rejeitados = {r.indice for r in resultado.rejeicoes}
    assert set(indices_violadores) <= indices_rejeitados  # ZERO falsos negativos
    # ZERO falsos positivos: toda a linha rejeitada viola mesmo a regra
    # (a distribuição normal de 'idade' também produz algumas violações
    # naturais além das injetadas — por isso comparamos por verificação
    # direta, não por igualdade exata de conjuntos de índices)
    for indice in indices_rejeitados:
        assert df.loc[indice, "idade"] < 18


# --------------------------------------------------------------------------
# Estatísticas suficientes acumuladas
# --------------------------------------------------------------------------


def test_running_covariance_accumulator_corresponde_ao_pandas_corr() -> None:
    df = pd.DataFrame(
        {
            "a": np.random.default_rng(1).normal(0, 1, 500),
            "b": np.random.default_rng(2).normal(0, 1, 500),
        }
    )
    df["b"] = 0.6 * df["a"] + df["b"]

    acumulador = RunningCovarianceAccumulator(["a", "b"])
    acumulador.update(df.iloc[:200])
    acumulador.update(df.iloc[200:])

    esperado = df.corr(method="pearson")
    obtido = acumulador.correlation_matrix()

    assert obtido.loc["a", "b"] == pytest.approx(esperado.loc["a", "b"], abs=1e-9)
    assert acumulador.n == 500


# --------------------------------------------------------------------------
# Frente 2 — Validação Causal Contínua
# --------------------------------------------------------------------------


def test_fisher_z_comparison_mesma_correlacao_nao_e_significativa() -> None:
    p = fisher_z_comparison_p_value(0.9, 1000, 0.9, 1000)
    assert p > 0.5


def test_fisher_z_comparison_correlacoes_muito_diferentes_e_significativa() -> None:
    p = fisher_z_comparison_p_value(0.9, 1000, 0.0, 1000)
    assert p < 0.01


def test_run_causal_monitoring_sem_deriva_quando_dados_seguem_o_contrato(
    contrato_idade_saldo_com_regra, dag_idade_saldo
) -> None:
    df = _gerar_dados_correlacionados(2000, rho=0.9, rng_seed=3)
    corr_empirica = df.corr(method="pearson")

    relatorio = run_causal_monitoring(
        contrato_idade_saldo_com_regra, dag_idade_saldo, corr_empirica, n_accumulated=2000
    )
    assert relatorio.derivas_detectadas == []


def test_run_causal_monitoring_deteta_correlacao_desviada(
    contrato_idade_saldo_com_regra, dag_idade_saldo
) -> None:
    df = _gerar_dados_correlacionados(2000, rho=0.0, rng_seed=4)  # correlação quebrada
    corr_empirica = df.corr(method="pearson")

    relatorio = run_causal_monitoring(
        contrato_idade_saldo_com_regra, dag_idade_saldo, corr_empirica, n_accumulated=2000
    )
    assert len(relatorio.derivas_detectadas) >= 1
    assert relatorio.derivas_detectadas[0].tipo == "correlacao_desviada"


def test_run_causal_monitoring_deteta_dependencia_emergente() -> None:
    # DAG assume educacao e risco d-separados dado rendimento; dados mostram
    # dependência residual forte (efeito direto não previsto pela DAG).
    dag = build_dag(
        ["educacao", "rendimento", "risco"],
        [("educacao", "rendimento"), ("rendimento", "risco")],
    )
    column_stats = {
        "educacao": _continuous(12, 3),
        "rendimento": _continuous(2000, 500),
        "risco": _continuous(0.3, 0.1),
    }
    contrato = _make_contract(
        column_stats,
        correlations=[
            CorrelationPair(coluna_a="educacao", coluna_b="rendimento", valor=0.6),
            CorrelationPair(coluna_a="rendimento", coluna_b="risco", valor=0.6),
            CorrelationPair(coluna_a="educacao", coluna_b="risco", valor=0.6),
        ],
    )
    cols = ["educacao", "risco", "rendimento"]
    valores = np.full((3, 3), 0.6)
    np.fill_diagonal(valores, 1.0)
    corr_empirica = pd.DataFrame(valores, index=cols, columns=cols)

    relatorio = run_causal_monitoring(contrato, dag, corr_empirica, n_accumulated=1000)
    tipos = {a.tipo for a in relatorio.derivas_detectadas}
    assert "dependencia_emergente" in tipos


# --------------------------------------------------------------------------
# Pilar 2 / Teste 1 — deteção de deriva causal ao longo de várias dezenas de milhar de linhas
# --------------------------------------------------------------------------


def test_run_hipotalamo_deteta_deriva_apos_varios_checkpoints(
    contrato_idade_saldo_com_regra, dag_idade_saldo
) -> None:
    """Simula um fluxo de lotes sucessivos (como viriam do Módulo 3): a
    correlação mantém-se fiel ao Contrato durante vários checkpoints e
    quebra subitamente perto do fim — o alerta de deriva deve surgir
    assim que o desvio se torna estatisticamente detetável, sem esperar
    até ao fim de todo o fluxo.
    """
    estado = HipotalamoState(contrato_idade_saldo_com_regra)
    alertas_fase_estavel: list[int] = []
    alertas_fase_deriva: list[int] = []

    for i in range(10):
        rho = 0.9 if i < 8 else 0.0  # quebra de correlação nos 2 últimos lotes
        lote = _gerar_dados_correlacionados(5000, rho=rho, rng_seed=100 + i)
        resultado = run_hipotalamo(
            lote, contrato_idade_saldo_com_regra, dag_idade_saldo, estado, checkpoint_size=5000
        )
        if resultado.monitorizacao_causal is not None:
            n_alertas = len(resultado.monitorizacao_causal.derivas_detectadas)
            (alertas_fase_estavel if i < 8 else alertas_fase_deriva).append(n_alertas)

    assert len(alertas_fase_estavel) >= 1  # pelo menos um checkpoint atingido na fase estável
    assert sum(alertas_fase_estavel) == 0  # nenhuma deriva enquanto a correlação se mantém
    assert len(alertas_fase_deriva) >= 1  # pelo menos um checkpoint atingido na fase de quebra
    assert sum(alertas_fase_deriva) > 0  # deriva detectada após a quebra de correlação


# --------------------------------------------------------------------------
# Frente 3 — Indução de Regras (ILP)
# --------------------------------------------------------------------------


@pytest.fixture
def contrato_e_dag_para_ilp():
    column_stats = {
        "rendimento": _continuous(3000, 1500),
        "idade": _continuous(40, 12),
        "risco": CategoricalStatistics(frequencias={"A": 0.5, "B": 0.5}),
    }
    contrato = _make_contract(column_stats, correlations=[])
    # já existe rendimento -> risco na DAG (a relação é um subproduto direto)
    dag = build_dag(["rendimento", "idade", "risco"], [("rendimento", "risco")])
    return contrato, dag


def _buffer_ilp(n: int = 500, rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rendimento = rng.normal(3000, 1500, n)
    idade = rng.normal(40, 12, n)
    # risco = B quase sempre que rendimento > 5000 (padrão determinístico)
    risco = np.where(rendimento > 5000, "B", "A")
    return pd.DataFrame({"rendimento": rendimento, "idade": idade, "risco": risco})


def _buffer_ilp_quantile(n: int = 2000, rng_seed: int = 0, quantil: float = 0.9) -> pd.DataFrame:
    """Buffer cujo padrão determinístico usa exatamente um dos quantis
    testados por `induce_rules` (ver `ILP_CANDIDATE_QUANTILES`) como
    limiar — garante confiança=1.0 nesse candidato específico, em vez
    de depender de o limiar de negócio coincidir por acaso com a grelha
    de quantis pesquisada.
    """
    rng = np.random.default_rng(rng_seed)
    rendimento = rng.normal(3000, 1500, n)
    idade = rng.normal(40, 12, n)
    limiar = np.quantile(rendimento, quantil)
    risco = np.where(rendimento > limiar, "B", "A")
    return pd.DataFrame({"rendimento": rendimento, "idade": idade, "risco": risco})


def test_induce_rules_descarta_subproduto_direto_da_dag(contrato_e_dag_para_ilp) -> None:
    contrato, dag = contrato_e_dag_para_ilp
    buffer = _buffer_ilp_quantile(n=2000, rng_seed=1)

    candidatas, descartadas_por_dag = induce_rules(buffer, contrato, dag, min_support=100, min_confidence=0.95)

    # nenhuma candidata proposta envolve 'rendimento -> risco' (já está na DAG)
    assert all(c.coluna_condicao != "rendimento" for c in candidatas)
    assert descartadas_por_dag > 0


def test_induce_rules_propoe_relacao_nao_presente_na_dag() -> None:
    column_stats = {
        "rendimento": _continuous(3000, 1500),
        "idade": _continuous(40, 12),
        "risco": CategoricalStatistics(frequencias={"A": 0.5, "B": 0.5}),
    }
    contrato = _make_contract(column_stats, correlations=[])
    dag_sem_aresta = build_dag(["rendimento", "idade", "risco"], [])  # SEM aresta rendimento->risco
    buffer = _buffer_ilp_quantile(n=2000, rng_seed=2)

    candidatas, descartadas = induce_rules(buffer, contrato, dag_sem_aresta, min_support=100, min_confidence=0.95)

    assert len(candidatas) > 0
    assert descartadas == 0
    assert any(c.coluna_condicao == "rendimento" and c.valor_alvo == "B" for c in candidatas)


def test_induce_rules_respeita_suporte_minimo() -> None:
    column_stats = {
        "rendimento": _continuous(3000, 1500),
        "idade": _continuous(40, 12),
        "risco": CategoricalStatistics(frequencias={"A": 0.9, "B": 0.1}),
    }
    contrato = _make_contract(column_stats, correlations=[])
    dag = build_dag(["rendimento", "idade", "risco"], [])
    buffer = _buffer_ilp(n=50, rng_seed=3)  # amostra pequena, suporte insuficiente

    candidatas, _ = induce_rules(buffer, contrato, dag, min_support=100, min_confidence=0.9)
    assert candidatas == []


def test_validate_candidate_with_asp_consolida_sem_excecoes() -> None:
    buffer = _buffer_ilp(n=1000, rng_seed=5)
    candidata = CandidateRule(
        coluna_condicao="rendimento",
        operador=RuleOperator.GT,
        limiar=5000.0,
        coluna_alvo="risco",
        valor_alvo="B",
        suporte=int((buffer["rendimento"] > 5000).sum()),
        confianca=1.0,
    )
    resultado = validate_candidate_with_asp(candidata, buffer)
    assert resultado.consolidada is True
    assert resultado.violacoes == 0


def test_validate_candidate_with_asp_rejeita_com_excecoes() -> None:
    buffer = _buffer_ilp(n=1000, rng_seed=6)
    # introduz exceções deliberadas
    buffer.loc[buffer.index[:5], "risco"] = "A"
    buffer.loc[buffer["rendimento"] > 5000, "risco"] = np.where(
        buffer.loc[buffer["rendimento"] > 5000].index % 50 == 0, "A", "B"
    )
    candidata = CandidateRule(
        coluna_condicao="rendimento",
        operador=RuleOperator.GT,
        limiar=5000.0,
        coluna_alvo="risco",
        valor_alvo="B",
        suporte=int((buffer["rendimento"] > 5000).sum()),
        confianca=0.9,
    )
    resultado = validate_candidate_with_asp(candidata, buffer)
    assert resultado.consolidada is False
    assert resultado.violacoes > 0
    assert "exceç" in resultado.motivo_rejeicao


def test_run_ilp_consolida_regra_valida_e_descarta_subproduto(contrato_e_dag_para_ilp) -> None:
    contrato, dag = contrato_e_dag_para_ilp
    buffer = _buffer_ilp_quantile(n=2000, rng_seed=7)

    relatorio, novas_regras = run_ilp(buffer, contrato, dag, min_support=100, min_confidence=0.95)
    assert relatorio.descartadas_por_dag > 0
    assert all(r.coluna_condicao != "rendimento" for r in novas_regras)


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def test_run_hipotalamo_lote_vazio_levanta_erro(contrato_idade_saldo_com_regra, dag_idade_saldo) -> None:
    estado = HipotalamoState(contrato_idade_saldo_com_regra)
    with pytest.raises(HipotalamoValidationError):
        run_hipotalamo(pd.DataFrame(columns=["idade", "saldo"]), contrato_idade_saldo_com_regra, dag_idade_saldo, estado)


def test_run_hipotalamo_sem_checkpoint_nao_faz_monitorizacao(
    contrato_idade_saldo_com_regra, dag_idade_saldo
) -> None:
    estado = HipotalamoState(contrato_idade_saldo_com_regra)
    lote_pequeno = _gerar_dados_correlacionados(100, rho=0.9, rng_seed=8)

    resultado = run_hipotalamo(
        lote_pequeno, contrato_idade_saldo_com_regra, dag_idade_saldo, estado, checkpoint_size=5000
    )
    assert resultado.monitorizacao_causal is None
    assert resultado.inducao is None


def test_run_hipotalamo_estado_acumula_entre_lotes(
    contrato_idade_saldo_com_regra, dag_idade_saldo
) -> None:
    estado = HipotalamoState(contrato_idade_saldo_com_regra)
    total_aprovadas_esperado = 0
    for i in range(3):
        lote = _gerar_dados_correlacionados(1000, rho=0.9, rng_seed=20 + i)
        resultado = run_hipotalamo(
            lote, contrato_idade_saldo_com_regra, dag_idade_saldo, estado, checkpoint_size=5000
        )
        total_aprovadas_esperado += resultado.validacao.aprovadas

    assert estado.total_aprovadas == total_aprovadas_esperado
    assert estado.accumulator.n == total_aprovadas_esperado
    assert estado.total_aprovadas > 2800  # a regra 'idade>=18' rejeita apenas uma pequena fração


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end
# --------------------------------------------------------------------------


def test_hipotalamo_end_to_end_pipeline_completo() -> None:
    """Testa o módulo como um todo: validação ASP + monitorização causal +
    ILP, ao longo de vários lotes sucessivos, com regras de negócio,
    correlações e uma relação induzível não presente na DAG.
    """
    column_stats = {
        "rendimento": _continuous(3000, 1500),
        "idade": _continuous(40, 12),
        "risco": CategoricalStatistics(frequencias={"A": 0.5, "B": 0.5}),
    }
    contrato = _make_contract(
        column_stats,
        correlations=[CorrelationPair(coluna_a="idade", coluna_b="rendimento", valor=0.0)],
        rules=[BusinessRule(coluna="rendimento", operador=RuleOperator.GT, valor=0.0, texto_original="rendimento > 0")],
    )
    dag = build_dag(["rendimento", "idade", "risco"], [])  # sem arestas: relação é nova

    estado = HipotalamoState(contrato)
    resultado_final = None
    for i in range(3):
        buffer = _buffer_ilp_quantile(n=3000, rng_seed=30 + i)
        buffer.loc[buffer.index[:3], "rendimento"] = -100  # viola 'rendimento > 0'
        buffer.loc[buffer.index[:3], "risco"] = "A"  # mantém consistência com o padrão induzido
        resultado_final = run_hipotalamo(buffer, contrato, dag, estado, checkpoint_size=3000, min_support_ilp=100)

    assert resultado_final is not None
    assert resultado_final.validacao.rejeitadas >= 3
    assert resultado_final.inducao is not None
    assert len(resultado_final.novas_regras) > 0
    assert any(r.coluna_condicao == "rendimento" and r.valor_alvo == "B" for r in resultado_final.novas_regras)
