"""Testes do módulo Neocórtex Gerador com Atenção Dinâmica e Rejeição de Estranheza."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synap.expansor import (
    ExpansorValidationError,
    InsufficientMemoryError,
    NegativeAttentionSignal,
    RejectionReason,
    apply_negative_attention,
    build_target_distribution,
    dag_column_groups,
    generate_chimera_row,
    mahalanobis_distance,
    penalize_sources,
    run_expansor,
    train_plausibility_model,
)
from synap.hipocampo.hipocampo import build_dag
from synap.neocortex.models import (
    CategoricalStatistics,
    ContinuousStatistics,
    ContractSource,
    CorrelationPair,
    DistributionType,
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


def _make_contract(column_stats: dict, correlations: list[CorrelationPair]) -> StatisticalContract:
    return StatisticalContract(
        column_stats=column_stats,
        correlations=correlations,
        tolerancia_kl=0.05,
        source=ContractSource.REAL_CSV,
        is_synthetic_pure=False,
        sample_size=1000,
        rules=[],
        warnings=[],
    )


@pytest.fixture
def contrato_idade_saldo() -> StatisticalContract:
    """Contrato com correlação forte (0.9) entre idade e saldo."""
    column_stats = {
        "idade": _continuous(45, 15),
        "saldo": _continuous(50_000, 20_000),
    }
    correlations = [CorrelationPair(coluna_a="idade", coluna_b="saldo", valor=0.9)]
    return _make_contract(column_stats, correlations)


def _seed_correlacionada(contrato: StatisticalContract, n: int, rng_seed: int = 0) -> pd.DataFrame:
    """Gera uma semente sintética respeitando a correlação idade-saldo do contrato."""
    rng = np.random.default_rng(rng_seed)
    idade_stat = contrato.column_stats["idade"]
    saldo_stat = contrato.column_stats["saldo"]
    rho = contrato.correlations[0].valor

    z_idade = rng.normal(0, 1, n)
    z_saldo = rho * z_idade + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    idade = idade_stat.media + idade_stat.desvio * z_idade
    saldo = saldo_stat.media + saldo_stat.desvio * z_saldo
    return pd.DataFrame({"idade": idade, "saldo": saldo})


# --------------------------------------------------------------------------
# build_target_distribution / mahalanobis_distance
# --------------------------------------------------------------------------


def test_build_target_distribution_forma_correta(contrato_idade_saldo) -> None:
    mean, inv_cov, cols = build_target_distribution(contrato_idade_saldo)
    assert cols == ["idade", "saldo"]
    assert mean == pytest.approx([45, 50_000])
    assert inv_cov.shape == (2, 2)


def test_build_target_distribution_sem_colunas_numericas() -> None:
    contrato = _make_contract(
        {"risco": CategoricalStatistics(frequencias={"baixo": 0.5, "alto": 0.5})}, []
    )
    mean, inv_cov, cols = build_target_distribution(contrato)
    assert cols == []
    assert mean.size == 0


def test_mahalanobis_distance_na_media_e_proxima_de_zero(contrato_idade_saldo) -> None:
    mean, inv_cov, cols = build_target_distribution(contrato_idade_saldo)
    d = mahalanobis_distance({"idade": 45, "saldo": 50_000}, mean, inv_cov, cols)
    assert d == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Pilar 1 / Teste 1 — filtro descarta > 90% das combinações que violam a correlação
# --------------------------------------------------------------------------


def test_mahalanobis_rejeita_mais_de_90_porcento_combinacoes_adversariais(
    contrato_idade_saldo,
) -> None:
    """Combinações cruzadas nos cantos opostos da distribuição conjunta
    (idade alta + saldo baixo, e vice-versa) violam fortemente a
    correlação de 0.9 esperada — o filtro deve rejeitar > 90% delas.
    """
    mean, inv_cov, cols = build_target_distribution(contrato_idade_saldo)
    idade_stat = contrato_idade_saldo.column_stats["idade"]
    saldo_stat = contrato_idade_saldo.column_stats["saldo"]

    magnitudes = [1.5, 2.0, 2.5, 3.0]
    linhas_adversariais = []
    for z in magnitudes:
        linhas_adversariais.append(
            {"idade": idade_stat.media + z * idade_stat.desvio, "saldo": saldo_stat.media - z * saldo_stat.desvio}
        )
        linhas_adversariais.append(
            {"idade": idade_stat.media - z * idade_stat.desvio, "saldo": saldo_stat.media + z * saldo_stat.desvio}
        )

    distancias = [mahalanobis_distance(linha, mean, inv_cov, cols) for linha in linhas_adversariais]
    fracao_rejeitada = np.mean([d > 3.0 for d in distancias])
    assert fracao_rejeitada > 0.90


# --------------------------------------------------------------------------
# Pilar 1 / Teste 2 — mantém linha rara mas realista (não descarta indevidamente)
# --------------------------------------------------------------------------


def test_mahalanobis_mantem_linha_rara_mas_consistente_com_correlacao(
    contrato_idade_saldo,
) -> None:
    """idade=70 (rara) com saldo no ponto ótimo da reta de regressão
    (consistente com a correlação de 0.9) não deve ser descartada,
    mesmo sendo uma combinação pouco frequente.
    """
    mean, inv_cov, cols = build_target_distribution(contrato_idade_saldo)
    idade_stat = contrato_idade_saldo.column_stats["idade"]
    saldo_stat = contrato_idade_saldo.column_stats["saldo"]
    rho = contrato_idade_saldo.correlations[0].valor

    z_idade = (70 - idade_stat.media) / idade_stat.desvio
    z_saldo_otimo = rho * z_idade  # ponto da reta de regressão, minimiza a distância
    saldo_consistente = saldo_stat.media + z_saldo_otimo * saldo_stat.desvio

    linha = {"idade": 70, "saldo": saldo_consistente}
    distancia = mahalanobis_distance(linha, mean, inv_cov, cols)
    assert distancia < 3.0


# --------------------------------------------------------------------------
# Remistura composicional
# --------------------------------------------------------------------------


def test_generate_chimera_row_usa_apenas_fontes_validas() -> None:
    memory = pd.DataFrame({"a": range(10), "b": range(10, 20)})
    rng = np.random.default_rng(0)
    weights = np.ones(10)
    linha, usados = generate_chimera_row(memory, weights, k=3, rng=rng)

    assert set(linha) == {"a", "b"}
    assert all(0 <= i < 10 for i in usados)
    assert linha["a"] in memory["a"].to_numpy()
    assert linha["b"] in memory["b"].to_numpy()


def test_generate_chimera_row_respeita_grupos_de_colunas() -> None:
    memory = pd.DataFrame({"a": [0, 1, 2], "b": [10, 11, 12], "c": [20, 21, 22]})
    rng = np.random.default_rng(1)
    weights = np.ones(3)
    grupos = [["a", "b"], ["c"]]

    for _ in range(20):
        linha, _ = generate_chimera_row(memory, weights, k=3, rng=rng, column_groups=grupos)
        # 'a' e 'b' vieram sempre da mesma linha-fonte (mesmo índice original)
        idx_correspondente = memory.index[(memory["a"] == linha["a"]) & (memory["b"] == linha["b"])]
        assert len(idx_correspondente) == 1


def test_dag_column_groups_agrupa_componentes_conexos() -> None:
    dag = build_dag(["a", "b", "c", "d"], [("a", "b")])
    grupos = dag_column_groups(dag, ["a", "b", "c", "d"])
    grupos_como_sets = {frozenset(g) for g in grupos}
    assert frozenset({"a", "b"}) in grupos_como_sets
    assert frozenset({"c"}) in grupos_como_sets
    assert frozenset({"d"}) in grupos_como_sets


def test_dag_column_groups_none_sem_dag() -> None:
    assert dag_column_groups(None, ["a", "b"]) is None


# --------------------------------------------------------------------------
# Penalização e atenção negativa
# --------------------------------------------------------------------------


def test_penalize_sources_reduz_peso_apenas_dos_indices_indicados() -> None:
    weights = np.ones(5)
    novos = penalize_sources(weights, [1, 3], factor=0.5)
    assert novos[1] == pytest.approx(0.5)
    assert novos[3] == pytest.approx(0.5)
    assert novos[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)  # não muta o original


def test_apply_negative_attention_penaliza_fonte_indicada() -> None:
    weights = np.ones(5)
    sinais = [NegativeAttentionSignal(indice_fonte=2, penalizacao=0.2)]
    novos, warnings = apply_negative_attention(weights, sinais)
    assert novos[2] == pytest.approx(0.2)
    assert warnings == []


def test_apply_negative_attention_indice_invalido_gera_aviso() -> None:
    weights = np.ones(3)
    sinais = [NegativeAttentionSignal(indice_fonte=99, penalizacao=0.2)]
    novos, warnings = apply_negative_attention(weights, sinais)
    assert len(warnings) == 1
    assert "fora da memória" in warnings[0]


# --------------------------------------------------------------------------
# Pilar 3 / Teste 3 — atenção negativa reduz recorrência de uma fonte em >= 60%
# --------------------------------------------------------------------------


def test_atencao_negativa_reduz_recorrencia_da_fonte_em_pelo_menos_60_porcento() -> None:
    memory = pd.DataFrame({"a": range(50), "b": range(50, 100), "c": range(100, 150)})
    n_iteracoes = 4000
    indice_alvo = 0

    def _frequencia_de_uso(weights: np.ndarray, rng_seed: int) -> float:
        rng = np.random.default_rng(rng_seed)
        contagem = 0
        for _ in range(n_iteracoes):
            _, usados = generate_chimera_row(memory, weights, k=5, rng=rng)
            if indice_alvo in usados:
                contagem += 1
        return contagem / n_iteracoes

    pesos_base = np.ones(50)
    frequencia_base = _frequencia_de_uso(pesos_base, rng_seed=1)

    sinais = [NegativeAttentionSignal(indice_fonte=indice_alvo, penalizacao=0.1)]
    pesos_penalizados, _ = apply_negative_attention(pesos_base, sinais)
    frequencia_penalizada = _frequencia_de_uso(pesos_penalizados, rng_seed=1)

    assert frequencia_base > 0.0  # sanidade: a fonte era de facto usada antes
    reducao = (frequencia_base - frequencia_penalizada) / frequencia_base
    assert reducao >= 0.60


# --------------------------------------------------------------------------
# Modelo de plausibilidade
# --------------------------------------------------------------------------


def test_plausibilidade_atribui_score_mais_alto_a_combinacoes_reais(
    contrato_idade_saldo,
) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=400, rng_seed=3)
    rng = np.random.default_rng(3)
    scorer = train_plausibility_model(seed, contrato_idade_saldo, rng, rng_seed=3)

    # em média, combinações reais tendem a pontuar mais alto que pares
    # aleatórios descorrelacionados
    scores_reais = [seed.iloc[i].to_dict() for i in range(20)]
    scores_reais = [scorer.score_row(r) for r in scores_reais]

    idades = seed["idade"].to_numpy()
    saldos_baralhados = np.random.default_rng(9).permutation(seed["saldo"].to_numpy())
    scores_quebrados = [
        scorer.score_row({"idade": idades[i], "saldo": saldos_baralhados[i]}) for i in range(20)
    ]

    assert np.mean(scores_reais) > np.mean(scores_quebrados)


# --------------------------------------------------------------------------
# Validação de input
# --------------------------------------------------------------------------


def test_run_expansor_semente_insuficiente_levanta_erro(contrato_idade_saldo) -> None:
    seed_pequena = _seed_correlacionada(contrato_idade_saldo, n=2)
    with pytest.raises(InsufficientMemoryError):
        run_expansor(seed_pequena, contrato_idade_saldo, n_rows=10)


def test_run_expansor_n_rows_invalido_levanta_erro(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=100)
    with pytest.raises(ExpansorValidationError):
        run_expansor(seed, contrato_idade_saldo, n_rows=0)


def test_run_expansor_k_range_invalido_levanta_erro(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=100)
    with pytest.raises(ExpansorValidationError):
        run_expansor(seed, contrato_idade_saldo, n_rows=10, k_range=(5, 3))


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end
# --------------------------------------------------------------------------


def test_run_expansor_end_to_end_gera_lote_valido_e_respeita_mahalanobis(
    contrato_idade_saldo,
) -> None:
    """Testa o módulo como um todo: semente -> remistura filtrada -> lote final."""
    seed = _seed_correlacionada(contrato_idade_saldo, n=300, rng_seed=5)

    resultado = run_expansor(
        seed,
        contrato_idade_saldo,
        n_rows=200,
        mahalanobis_threshold=3.0,
        plausibility_threshold=0.05,
        rng_seed=42,
    )

    assert resultado.relatorio.linhas_geradas == 200
    assert len(resultado.lote) == 200
    assert set(resultado.lote.columns) == {"idade", "saldo"}

    mean, inv_cov, cols = build_target_distribution(contrato_idade_saldo)
    distancias = [
        mahalanobis_distance(row.to_dict(), mean, inv_cov, cols)
        for _, row in resultado.lote.iterrows()
    ]
    # invariante central do Pilar 1: NENHUMA linha aprovada viola o limiar
    assert all(d <= 3.0 for d in distancias)
    assert resultado.tamanho_memoria_final == len(seed) + 200


def test_run_expansor_com_dag_agrupa_colunas_ligadas(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=200, rng_seed=6)
    dag = build_dag(["idade", "saldo"], [("idade", "saldo")])

    resultado = run_expansor(
        seed, contrato_idade_saldo, n_rows=50, dag=dag, plausibility_threshold=0.02, rng_seed=1
    )

    assert len(resultado.lote) == 50


def test_run_expansor_relatorio_contabiliza_rejeicoes(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=200, rng_seed=7)
    resultado = run_expansor(seed, contrato_idade_saldo, n_rows=30, rng_seed=2)

    assert resultado.relatorio.tentativas_totais >= resultado.relatorio.linhas_geradas
    assert set(resultado.relatorio.rejeicoes_por_motivo) == {r.value for r in RejectionReason}
