"""Testes do módulo Hipocampo e Motor de Génese Causal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synap.hipocampo import (
    CyclicGraphError,
    HipocampoValidationError,
    add_missing_edges,
    build_dag,
    compute_dag_bic,
    detect_missing_edges,
    fisher_z_p_value,
    generate_seed,
    learn_dag_from_contract,
    partial_correlation,
    run_hipocampo,
)
from synap.hipocampo.hipocampo import MAX_SEED_ROWS
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


def _continuous(media: float, desvio: float, minimo: float | None = None, maximo: float | None = None) -> ContinuousStatistics:
    minimo = media - 4 * desvio if minimo is None else minimo
    maximo = media + 4 * desvio if maximo is None else maximo
    return ContinuousStatistics(
        tipo=DistributionType.NORMAL,
        media=media,
        desvio=desvio,
        minimo=minimo,
        maximo=maximo,
        quartis=(media - 0.674 * desvio, media, media + 0.674 * desvio),
    )


def _categorical(frequencias: dict[str, float]) -> CategoricalStatistics:
    return CategoricalStatistics(frequencias=frequencias)


def _make_contract(
    column_stats: dict,
    correlations: list[CorrelationPair],
    sample_size: int | None = 1000,
    rules: list[BusinessRule] | None = None,
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
def contrato_mediado_com_efeito_direto() -> StatisticalContract:
    """Educação->Rendimento->Risco, mas com efeito direto residual Educação->Risco.

    Correlações: edu-rend=0.6, rend-risco=0.6, edu-risco=0.6 (muito acima
    do produto 0.36 esperado por mediação pura) -> a correlação parcial
    edu-risco|rend é fortemente significativa (~0.375, p << 0.01).
    """
    column_stats = {
        "educacao": _continuous(12, 3),
        "rendimento": _continuous(2000, 500),
        "risco": _continuous(0.3, 0.1),
    }
    correlations = [
        CorrelationPair(coluna_a="educacao", coluna_b="rendimento", valor=0.6),
        CorrelationPair(coluna_a="rendimento", coluna_b="risco", valor=0.6),
        CorrelationPair(coluna_a="educacao", coluna_b="risco", valor=0.6),
    ]
    return _make_contract(column_stats, correlations, sample_size=1000)


@pytest.fixture
def contrato_totalmente_mediado() -> StatisticalContract:
    """Educação->Rendimento->Risco com edu-risco EXATAMENTE = produto das arestas.

    edu-rend=0.5, rend-risco=0.5, edu-risco=0.25=0.5*0.5 -> correlação
    parcial edu-risco|rend é ~0 (mediação perfeita, sem efeito direto).
    """
    column_stats = {
        "educacao": _continuous(12, 3),
        "rendimento": _continuous(2000, 500),
        "risco": _continuous(0.3, 0.1),
    }
    correlations = [
        CorrelationPair(coluna_a="educacao", coluna_b="rendimento", valor=0.5),
        CorrelationPair(coluna_a="rendimento", coluna_b="risco", valor=0.5),
        CorrelationPair(coluna_a="educacao", coluna_b="risco", valor=0.25),
    ]
    return _make_contract(column_stats, correlations, sample_size=1000)


@pytest.fixture
def dag_edu_rend_risco():
    return build_dag(
        ["educacao", "rendimento", "risco"],
        [("educacao", "rendimento"), ("rendimento", "risco")],
    )


# --------------------------------------------------------------------------
# CausalDAG / build_dag
# --------------------------------------------------------------------------


def test_build_dag_valido(dag_edu_rend_risco) -> None:
    assert dag_edu_rend_risco.has_edge("educacao", "rendimento")
    assert dag_edu_rend_risco.parents("risco") == ["rendimento"]
    assert dag_edu_rend_risco.topological_order()[0] == "educacao"


def test_build_dag_ciclica_levanta_erro() -> None:
    with pytest.raises(CyclicGraphError):
        build_dag(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])


def test_build_dag_no_inexistente_levanta_erro() -> None:
    with pytest.raises(HipocampoValidationError):
        build_dag(["a", "b"], [("a", "z")])


def test_causal_dag_e_imutavel(dag_edu_rend_risco) -> None:
    with pytest.raises(Exception):
        dag_edu_rend_risco.nos = ["outro"]  # type: ignore[misc]


def test_causal_dag_with_edge_produz_nova_instancia(dag_edu_rend_risco) -> None:
    nova = dag_edu_rend_risco.with_edge("educacao", "risco")
    assert nova.has_edge("educacao", "risco")
    assert not dag_edu_rend_risco.has_edge("educacao", "risco")  # original inalterada


def test_causal_dag_with_edge_ciclica_levanta_erro(dag_edu_rend_risco) -> None:
    with pytest.raises(Exception):
        dag_edu_rend_risco.with_edge("risco", "educacao")


def test_causal_dag_has_directed_path(dag_edu_rend_risco) -> None:
    assert dag_edu_rend_risco.has_directed_path("educacao", "risco")
    assert not dag_edu_rend_risco.has_directed_path("risco", "educacao")


# --------------------------------------------------------------------------
# Correlação parcial e Fisher-Z
# --------------------------------------------------------------------------


def test_partial_correlation_sem_condicionamento_e_correlacao_direta() -> None:
    matriz = pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])
    assert partial_correlation(matriz, "a", "b", []) == pytest.approx(0.5)


def test_partial_correlation_equicorrelacionada_valor_conhecido() -> None:
    # Matriz de equicorrelação 3x3 com rho=0.6 -> partial(x,y|z) = 0.375 (calculado analiticamente)
    cols = ["educacao", "risco", "rendimento"]
    valores = np.full((3, 3), 0.6)
    np.fill_diagonal(valores, 1.0)
    matriz = pd.DataFrame(valores, index=cols, columns=cols)
    rho = partial_correlation(matriz, "educacao", "risco", ["rendimento"])
    assert rho == pytest.approx(0.375, abs=1e-3)


def test_partial_correlation_mediacao_perfeita_e_zero() -> None:
    matriz = pd.DataFrame(
        {
            "educacao": [1.0, 0.25, 0.5],
            "risco": [0.25, 1.0, 0.5],
            "rendimento": [0.5, 0.5, 1.0],
        },
        index=["educacao", "risco", "rendimento"],
    )
    rho = partial_correlation(matriz, "educacao", "risco", ["rendimento"])
    assert rho == pytest.approx(0.0, abs=1e-9)


def test_fisher_z_p_value_correlacao_forte_e_significativa() -> None:
    p = fisher_z_p_value(0.375, n=1000, cond_set_size=1)
    assert p < 0.01


def test_fisher_z_p_value_correlacao_nula_nao_e_significativa() -> None:
    p = fisher_z_p_value(0.0, n=1000, cond_set_size=1)
    assert p > 0.5


def test_fisher_z_p_value_graus_liberdade_insuficientes_devolve_um() -> None:
    assert fisher_z_p_value(0.9, n=3, cond_set_size=5) == 1.0


# --------------------------------------------------------------------------
# Pilar 2 / Teste 1 — deteção e adição de aresta em falta, melhoria BIC >= 20%
# --------------------------------------------------------------------------


def test_detect_missing_edges_encontra_aresta_edu_risco(
    dag_edu_rend_risco, contrato_mediado_com_efeito_direto
) -> None:
    candidatos = detect_missing_edges(dag_edu_rend_risco, contrato_mediado_com_efeito_direto)

    assert len(candidatos) == 1
    ajuste = candidatos[0]
    assert {ajuste.origem, ajuste.destino} == {"educacao", "risco"}
    assert ajuste.origem == "educacao"  # direção inferida do caminho edu->rend->risco
    assert ajuste.p_valor < 0.01
    assert ajuste.melhoria_percentual >= 0.20


def test_add_missing_edges_aplica_aresta_e_relata_melhoria(
    dag_edu_rend_risco, contrato_mediado_com_efeito_direto
) -> None:
    dag_ajustada, ajustes, warnings = add_missing_edges(
        dag_edu_rend_risco, contrato_mediado_com_efeito_direto
    )

    assert dag_ajustada.has_edge("educacao", "risco")
    assert len(ajustes) == 1
    assert ajustes[0].melhoria_percentual >= 0.20
    assert any("aresta em falta detetada" in w for w in warnings)


def test_compute_dag_bic_melhora_apos_adicionar_aresta(
    dag_edu_rend_risco, contrato_mediado_com_efeito_direto
) -> None:
    # A garantia de melhoria >= 20% (Pilar 2 / Teste 1) aplica-se ao ajuste
    # LOCAL do nó destino (ver DAGAdjustment.melhoria_percentual, testado
    # em test_detect_missing_edges_encontra_aresta_edu_risco). O BIC GLOBAL
    # do DAG soma também nós inalterados (ex.: rendimento|educacao), pelo
    # que a melhoria percentual global é necessariamente menor — o
    # invariante verificável aqui é que o ajuste global estritamente
    # melhora (bic_depois < bic_antes), nunca piora.
    bic_antes = compute_dag_bic(dag_edu_rend_risco, contrato_mediado_com_efeito_direto)
    dag_com_aresta = dag_edu_rend_risco.with_edge("educacao", "risco")
    bic_depois = compute_dag_bic(dag_com_aresta, contrato_mediado_com_efeito_direto)

    assert bic_depois < bic_antes
    melhoria_global = (bic_antes - bic_depois) / abs(bic_antes)
    assert melhoria_global > 0.0


def test_compute_dag_bic_sem_sample_size_levanta_erro(dag_edu_rend_risco, contrato_mediado_com_efeito_direto) -> None:
    contrato_sem_n = contrato_mediado_com_efeito_direto.model_copy(update={"sample_size": None})
    with pytest.raises(HipocampoValidationError):
        compute_dag_bic(dag_edu_rend_risco, contrato_sem_n)


# --------------------------------------------------------------------------
# Pilar 2 / Teste 2 — não adicionar arestas espúrias (mediação perfeita)
# --------------------------------------------------------------------------


def test_detect_missing_edges_nao_adiciona_aresta_espuria(
    dag_edu_rend_risco, contrato_totalmente_mediado
) -> None:
    candidatos = detect_missing_edges(dag_edu_rend_risco, contrato_totalmente_mediado)
    assert candidatos == []


def test_add_missing_edges_dag_inalterada_quando_mediacao_perfeita(
    dag_edu_rend_risco, contrato_totalmente_mediado
) -> None:
    dag_ajustada, ajustes, _ = add_missing_edges(dag_edu_rend_risco, contrato_totalmente_mediado)
    assert ajustes == []
    assert not dag_ajustada.has_edge("educacao", "risco")
    assert not dag_ajustada.has_edge("risco", "educacao")


# --------------------------------------------------------------------------
# Pilar 2 / Teste 3 — semente com correlações fiéis (erro médio < 5%)
# --------------------------------------------------------------------------


def test_generate_seed_correlacoes_proximas_do_contrato(
    dag_edu_rend_risco, contrato_mediado_com_efeito_direto
) -> None:
    dag_final = dag_edu_rend_risco.with_edge("educacao", "risco")
    df, relatorio, warnings = generate_seed(
        dag_final, contrato_mediado_com_efeito_direto, n_rows=8000, rng_seed=42
    )

    assert len(df) == 8000
    assert set(df.columns) == {"educacao", "rendimento", "risco"}
    assert relatorio.correlacoes_avaliadas == 3
    assert relatorio.erro_medio_absoluto < 0.05


def test_generate_seed_n_rows_fora_do_intervalo_levanta_erro(
    dag_edu_rend_risco, contrato_mediado_com_efeito_direto
) -> None:
    with pytest.raises(HipocampoValidationError):
        generate_seed(dag_edu_rend_risco, contrato_mediado_com_efeito_direto, n_rows=100)
    with pytest.raises(HipocampoValidationError):
        generate_seed(dag_edu_rend_risco, contrato_mediado_com_efeito_direto, n_rows=MAX_SEED_ROWS + 1)


def test_generate_seed_respeita_regra_de_negocio(dag_edu_rend_risco) -> None:
    column_stats = {
        "educacao": _continuous(12, 3),
        "rendimento": _continuous(100, 500),  # média baixa perto de 0 -> muitas violações de ">0"
        "risco": _continuous(0.3, 0.1),
    }
    correlations = [
        CorrelationPair(coluna_a="educacao", coluna_b="rendimento", valor=0.4),
        CorrelationPair(coluna_a="rendimento", coluna_b="risco", valor=0.3),
        CorrelationPair(coluna_a="educacao", coluna_b="risco", valor=0.1),
    ]
    rules = [
        BusinessRule(
            coluna="rendimento", operador=RuleOperator.GT, valor=0.0, texto_original="rendimento > 0"
        )
    ]
    contrato = _make_contract(column_stats, correlations, sample_size=1000, rules=rules)

    df, relatorio, _ = generate_seed(dag_edu_rend_risco, contrato, n_rows=5000, rng_seed=1)

    assert (df["rendimento"] > 0).all()
    assert relatorio.regras_aplicadas == 1


def test_generate_seed_com_coluna_categorica_e_pai_numerico() -> None:
    dag = build_dag(["idade", "risco"], [("idade", "risco")])
    column_stats = {
        "idade": _continuous(40, 10),
        "risco": _categorical({"baixo": 0.5, "medio": 0.3, "alto": 0.2}),
    }
    contrato = _make_contract(column_stats, correlations=[], sample_size=1000)

    df, _, _ = generate_seed(dag, contrato, n_rows=5000, rng_seed=7)

    assert set(df["risco"].unique()) <= {"baixo", "medio", "alto"}
    assert len(df) == 5000


# --------------------------------------------------------------------------
# Aprendizagem de DAG do zero
# --------------------------------------------------------------------------


def test_learn_dag_from_contract_recupera_esqueleto_da_cadeia(
    contrato_totalmente_mediado,
) -> None:
    dag, warnings = learn_dag_from_contract(contrato_totalmente_mediado)

    # A cadeia educacao-rendimento-risco deve manter as arestas adjacentes;
    # a aresta direta educacao-risco não deve sobreviver (mediação perfeita).
    assert dag.is_adjacent("educacao", "rendimento")
    assert dag.is_adjacent("rendimento", "risco")
    assert not dag.is_adjacent("educacao", "risco")
    assert any("PC-algorithm" in w for w in warnings)


def test_learn_dag_from_contract_sem_sample_size_devolve_dag_vazia() -> None:
    column_stats = {"a": _continuous(0, 1), "b": _continuous(0, 1)}
    contrato = _make_contract(
        column_stats,
        correlations=[CorrelationPair(coluna_a="a", coluna_b="b", valor=0.5)],
        sample_size=None,
    )
    dag, warnings = learn_dag_from_contract(contrato)
    assert dag.arestas == []
    assert any("sample_size" in w for w in warnings)


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end
# --------------------------------------------------------------------------


def test_generate_seed_categorica_respeita_ordem_categorias_do_driver() -> None:
    """Regressão: sem usar `ordem_categorias`, o percentil do driver
    numérico era mapeado para categorias na ordem arbitrária do dicionário
    de frequências — podendo inverter/baralhar a direção real da relação
    (ex.: "alto risco" a corresponder a scores altos, ao contrário do real).
    """
    dag = build_dag(["score", "risco"], [("score", "risco")])
    column_stats = {
        "score": _continuous(650, 90),
        # ordem_categorias diz que 'alto' corresponde aos scores mais
        # baixos e 'baixo' aos mais altos — o oposto da ordem alfabética
        # e da ordem de inserção em frequencias
        "risco": CategoricalStatistics(
            frequencias={"alto": 0.2, "medio": 0.3, "baixo": 0.5},
            ordem_categorias=["alto", "medio", "baixo"],
        ),
    }
    contrato = _make_contract(column_stats, correlations=[])

    df, _, _ = generate_seed(dag, contrato, n_rows=5000, rng_seed=3)

    media_por_categoria = df.groupby("risco")["score"].mean()
    # 'alto' (primeiro em ordem_categorias, percentis mais baixos) tem de
    # corresponder à média de score MAIS BAIXA
    assert media_por_categoria["alto"] < media_por_categoria["medio"] < media_por_categoria["baixo"]


def test_generate_seed_categorica_sem_ordem_gera_aviso() -> None:
    dag = build_dag(["score", "risco"], [("score", "risco")])
    column_stats = {
        "score": _continuous(650, 90),
        "risco": CategoricalStatistics(frequencias={"alto": 0.2, "medio": 0.3, "baixo": 0.5}),
    }
    contrato = _make_contract(column_stats, correlations=[])

    _, _, warnings = generate_seed(dag, contrato, n_rows=5000, rng_seed=4)

    assert any("sem ordem de categorias detectada" in w for w in warnings)


def test_run_hipocampo_end_to_end_com_dag_do_utilizador(
    contrato_mediado_com_efeito_direto,
) -> None:
    """Testa o módulo como um todo: Contrato + DAG -> ajuste -> semente final."""
    resultado = run_hipocampo(
        contrato_mediado_com_efeito_direto,
        user_dag_edges=[("educacao", "rendimento"), ("rendimento", "risco")],
        n_rows=6000,
        rng_seed=99,
    )

    assert resultado.dag.has_edge("educacao", "risco")  # aresta em falta foi adicionada
    assert len(resultado.ajustes) == 1
    assert len(resultado.seed) == 6000
    assert resultado.relatorio.erro_medio_absoluto < 0.05
    assert any("aresta em falta detetada" in w for w in resultado.warnings)


def test_run_hipocampo_sem_dag_aprende_automaticamente(
    contrato_totalmente_mediado,
) -> None:
    resultado = run_hipocampo(contrato_totalmente_mediado, user_dag_edges=None, n_rows=5000)

    assert len(resultado.seed) == 5000
    assert resultado.dag.is_adjacent("educacao", "rendimento")
    assert resultado.dag.is_adjacent("rendimento", "risco")


def test_run_hipocampo_dag_com_no_fora_do_contrato_levanta_erro(
    contrato_mediado_com_efeito_direto,
) -> None:
    with pytest.raises(HipocampoValidationError):
        run_hipocampo(
            contrato_mediado_com_efeito_direto,
            user_dag_edges=[("educacao", "coluna_inexistente")],
        )
