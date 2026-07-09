"""Testes do módulo Núcleo Cognitivo de Dupla Via com Deteção de Artefactos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synap.expansor.expansor import generate_chimera_row, penalize_sources
from synap.hipocampo.hipocampo import build_dag
from synap.neocortex.models import (
    BusinessRule,
    ContinuousStatistics,
    ContractSource,
    CorrelationPair,
    DistributionType,
    RuleOperator,
    StatisticalContract,
)
from synap.nucleo import (
    InsufficientDataError,
    NucleoValidationError,
    run_nucleo,
    train_autoencoder,
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


def _make_contract(column_stats: dict, correlations: list[CorrelationPair], rules=None) -> StatisticalContract:
    return StatisticalContract(
        column_stats=column_stats,
        correlations=correlations,
        tolerancia_kl=0.05,
        source=ContractSource.REAL_CSV,
        is_synthetic_pure=False,
        sample_size=1000,
        rules=rules or [],
        warnings=[],
    )


@pytest.fixture
def contrato_idade_saldo() -> StatisticalContract:
    column_stats = {"idade": _continuous(45, 15), "saldo": _continuous(50_000, 20_000)}
    correlations = [CorrelationPair(coluna_a="idade", coluna_b="saldo", valor=0.9)]
    return _make_contract(column_stats, correlations)


def _seed_correlacionada(contrato: StatisticalContract, n: int, rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    idade_stat = contrato.column_stats["idade"]
    saldo_stat = contrato.column_stats["saldo"]
    rho = contrato.correlations[0].valor
    z1 = rng.normal(0, 1, n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "idade": idade_stat.media + idade_stat.desvio * z1,
            "saldo": saldo_stat.media + saldo_stat.desvio * z2,
        }
    )


# --------------------------------------------------------------------------
# train_autoencoder — validação de input
# --------------------------------------------------------------------------


def test_train_autoencoder_semente_insuficiente_levanta_erro(contrato_idade_saldo) -> None:
    seed_pequena = _seed_correlacionada(contrato_idade_saldo, n=10)
    with pytest.raises(InsufficientDataError):
        train_autoencoder(seed_pequena, contrato_idade_saldo)


# --------------------------------------------------------------------------
# Pilar 3 / Teste 1 — autoencoder identifica >= 95% de combinações raras
# --------------------------------------------------------------------------


def test_autoencoder_identifica_pelo_menos_95_porcento_das_combinacoes_raras(
    contrato_idade_saldo,
) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=500, rng_seed=1)
    scorer = train_autoencoder(seed, contrato_idade_saldo, rng_seed=1)

    idade_stat = contrato_idade_saldo.column_stats["idade"]
    saldo_stat = contrato_idade_saldo.column_stats["saldo"]

    # combinações cruzadas nos cantos opostos da distribuição conjunta:
    # violam fortemente a correlação de 0.9 esperada
    magnitudes = [1.5, 2.0, 2.5, 3.0]
    outliers = []
    for z in magnitudes:
        outliers.append(
            {"idade": idade_stat.media + z * idade_stat.desvio, "saldo": saldo_stat.media - z * saldo_stat.desvio}
        )
        outliers.append(
            {"idade": idade_stat.media - z * idade_stat.desvio, "saldo": saldo_stat.media + z * saldo_stat.desvio}
        )

    deteccoes = [scorer.is_strange(o) for o in outliers]
    assert np.mean(deteccoes) >= 0.95


# --------------------------------------------------------------------------
# Pilar 3 / Teste 2 — não classifica como estranha uma combinação rara mas plausível
# --------------------------------------------------------------------------


def test_autoencoder_nao_marca_combinacao_rara_mas_plausivel(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=500, rng_seed=2)
    scorer = train_autoencoder(seed, contrato_idade_saldo, rng_seed=2)

    idade_stat = contrato_idade_saldo.column_stats["idade"]
    saldo_stat = contrato_idade_saldo.column_stats["saldo"]
    rho = contrato_idade_saldo.correlations[0].valor

    # "bilionário com idade avançada": raro em termos absolutos, mas
    # consistente com a correlação idade-saldo esperada (ponto ótimo
    # da reta de regressão, minimiza a perda de reconstrução)
    z_idade = 2.0
    z_saldo_consistente = rho * z_idade
    linha_rara_mas_plausivel = {
        "idade": idade_stat.media + z_idade * idade_stat.desvio,
        "saldo": saldo_stat.media + z_saldo_consistente * saldo_stat.desvio,
    }

    assert not scorer.is_strange(linha_rara_mas_plausivel)


# --------------------------------------------------------------------------
# train_autoencoder — reciclagem incremental
# --------------------------------------------------------------------------


def test_autoencoder_partial_update_nao_falha_e_preserva_baseline(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=500, rng_seed=3)
    scorer = train_autoencoder(seed, contrato_idade_saldo, rng_seed=3)
    limiar_antes = scorer.baseline.limiar

    novas_linhas = _seed_correlacionada(contrato_idade_saldo, n=50, rng_seed=4)
    scorer.partial_update(novas_linhas)

    # a linha de base é deliberadamente fixa (ver docstring de partial_update)
    assert scorer.baseline.limiar == limiar_antes


# --------------------------------------------------------------------------
# Validações de input do run_nucleo
# --------------------------------------------------------------------------


def test_run_nucleo_semente_insuficiente_levanta_erro(contrato_idade_saldo) -> None:
    seed_pequena = _seed_correlacionada(contrato_idade_saldo, n=10)
    with pytest.raises(InsufficientDataError):
        run_nucleo(seed_pequena, contrato_idade_saldo, n_rows=10)


def test_run_nucleo_n_rows_invalido_levanta_erro(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=200)
    with pytest.raises(NucleoValidationError):
        run_nucleo(seed, contrato_idade_saldo, n_rows=0)


def test_run_nucleo_pesos_iniciais_com_tamanho_errado_levanta_erro(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=200)
    with pytest.raises(NucleoValidationError):
        run_nucleo(seed, contrato_idade_saldo, n_rows=10, initial_weights=np.ones(5))


# --------------------------------------------------------------------------
# Pilar 3 / Teste 3 — atenção negativa reduz a ocorrência de artefactos
# --------------------------------------------------------------------------


def test_atencao_negativa_reduz_ocorrencia_de_artefactos_apos_ciclos_de_ajuste(
    contrato_idade_saldo,
) -> None:
    """Uma fonte "envenenada" (combinação idade-saldo extrema, fora da
    variedade aprendida pelo autoencoder) é deliberadamente inserida na
    memória. Primeiro confirma-se que a sua utilização produz
    fiavelmente artefactos; depois mede-se, com uma amostra grande (para
    baixa variância), a frequência de seleção dessa fonte antes e
    depois de 3 ciclos de penalização — cada ciclo replica exatamente a
    penalização que `run_nucleo` aplica em tempo real sempre que um
    artefacto é detectado.
    """
    seed = _seed_correlacionada(contrato_idade_saldo, n=300, rng_seed=5)
    scorer = train_autoencoder(seed, contrato_idade_saldo, rng_seed=5)

    memoria = seed.sample(29, random_state=1).reset_index(drop=True)
    indice_veneno = len(memoria)
    memoria.loc[indice_veneno] = {"idade": 90, "saldo": -400_000}  # combinação extrema

    # Sanidade: usar o veneno como fonte produz artefactos fiavelmente
    # (força a sua inclusão nas k fontes para isolar o efeito)
    rng_sanidade = np.random.default_rng(11)
    pesos_forcados = np.zeros(len(memoria))
    pesos_forcados[indice_veneno] = 1.0
    pesos_forcados[:indice_veneno] = 1.0
    deteccoes = []
    for _ in range(100):
        linha, usados = generate_chimera_row(memoria, pesos_forcados, k=5, rng=rng_sanidade)
        if indice_veneno in usados:
            deteccoes.append(scorer.is_strange(linha))
    assert np.mean(deteccoes) >= 0.80  # o veneno gera artefacto na esmagadora maioria das vezes

    # Medição da frequência de seleção (amostra grande, baixa variância)
    def _frequencia_de_uso(pesos: np.ndarray, rng_seed: int, n_iter: int = 8000) -> float:
        rng = np.random.default_rng(rng_seed)
        contagem = 0
        for _ in range(n_iter):
            _, usados = generate_chimera_row(memoria, pesos, k=5, rng=rng)
            if indice_veneno in usados:
                contagem += 1
        return contagem / n_iter

    pesos_base = np.ones(len(memoria))
    frequencia_inicial = _frequencia_de_uso(pesos_base, rng_seed=1)

    # 3 ciclos de ajuste: cada um replica a penalização de artefacto
    # real (`run_nucleo` usa exatamente `penalize_sources` com
    # `artifact_penalty` sempre que `scorer.is_strange` é True)
    pesos_apos_ciclos = pesos_base.copy()
    for _ in range(3):
        pesos_apos_ciclos = penalize_sources(pesos_apos_ciclos, [indice_veneno], factor=0.3)
    frequencia_final = _frequencia_de_uso(pesos_apos_ciclos, rng_seed=1)

    assert frequencia_inicial > 0.0
    reducao = (frequencia_inicial - frequencia_final) / frequencia_inicial
    assert reducao >= 0.60


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end
# --------------------------------------------------------------------------


def test_run_nucleo_end_to_end_gera_lote_valido(contrato_idade_saldo) -> None:
    """Testa o módulo como um todo: semente -> ciclo neural-simbólico completo -> lote final."""
    regra = BusinessRule(coluna="saldo", operador=RuleOperator.GT, valor=-100_000, texto_original="saldo > -100000")
    contrato = contrato_idade_saldo.model_copy(update={"rules": [regra]})
    seed = _seed_correlacionada(contrato, n=300, rng_seed=6)

    resultado = run_nucleo(seed, contrato, n_rows=100, plausibility_threshold=0.02, rng_seed=42)

    assert resultado.relatorio.linhas_geradas == 100
    assert len(resultado.lote) == 100
    assert set(resultado.lote.columns) == {"idade", "saldo"}
    # nenhuma linha aprovada viola a regra de negócio (validação ASP funcionou)
    assert (resultado.lote["saldo"] > -100_000).all()
    assert resultado.memoria_final.shape[0] >= len(seed) + 100
    assert len(resultado.pesos_finais) == resultado.memoria_final.shape[0]


def test_run_nucleo_com_dag_agrupa_colunas(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=200, rng_seed=7)
    dag = build_dag(["idade", "saldo"], [("idade", "saldo")])

    resultado = run_nucleo(seed, contrato_idade_saldo, n_rows=30, dag=dag, plausibility_threshold=0.02, rng_seed=1)
    assert len(resultado.lote) == 30


def test_run_nucleo_encadeamento_de_ciclos_com_pesos_finais(contrato_idade_saldo) -> None:
    """Confirma que a memória e os pesos finais de um ciclo podem
    alimentar diretamente o ciclo seguinte (encadeamento de lotes)."""
    seed = _seed_correlacionada(contrato_idade_saldo, n=200, rng_seed=8)

    resultado1 = run_nucleo(seed, contrato_idade_saldo, n_rows=20, rng_seed=1)
    resultado2 = run_nucleo(
        resultado1.memoria_final,
        contrato_idade_saldo,
        n_rows=20,
        initial_weights=np.array(resultado1.pesos_finais),
        rng_seed=2,
    )

    assert len(resultado2.lote) == 20
    assert resultado2.memoria_final.shape[0] >= resultado1.memoria_final.shape[0] + 20


def test_run_nucleo_relatorio_contabiliza_artefactos(contrato_idade_saldo) -> None:
    seed = _seed_correlacionada(contrato_idade_saldo, n=200, rng_seed=9)
    resultado = run_nucleo(seed, contrato_idade_saldo, n_rows=30, rng_seed=3)

    assert resultado.relatorio.artefactos_detectados >= 0
    assert resultado.relatorio.tentativas_totais >= resultado.relatorio.linhas_geradas
