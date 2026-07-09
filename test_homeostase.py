"""Testes do módulo Controlador de Homeostasia e Qualidade Distribucional."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from synap.homeostase import (
    HealthStatus,
    HomeostaseState,
    HomeostaseValidationError,
    find_underrepresented_categories,
    gaussian_kl,
    normalized_divergence_score,
    run_controlador_homeostasia,
    shannon_entropy_normalized,
)
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


def _make_contract(column_stats: dict, correlations: list[CorrelationPair] | None = None) -> StatisticalContract:
    return StatisticalContract(
        column_stats=column_stats,
        correlations=correlations or [],
        tolerancia_kl=0.05,
        source=ContractSource.REAL_CSV,
        is_synthetic_pure=False,
        sample_size=1000,
        rules=[],
        warnings=[],
    )


@pytest.fixture
def contrato_rendimento() -> StatisticalContract:
    return _make_contract({"rendimento": _continuous(5000, 1500)})


def _pool_gaussiano(contrato: StatisticalContract, n: int, rng_seed: int) -> pd.DataFrame:
    """Pool de fontes com a mesma distribuição do Contrato (para simulação de correção)."""
    stat = contrato.column_stats["rendimento"]
    rng = np.random.default_rng(rng_seed)
    valores = rng.normal(stat.media, stat.desvio, n)
    return pd.DataFrame({"rendimento": valores[valores > 0]}).reset_index(drop=True)


def _run_correction_loop(
    contrato: StatisticalContract,
    pool: pd.DataFrame,
    vies_media: float,
    vies_desvio: float,
    n_checkpoints: int,
    batch_size: int = 5000,
    rng_seed: int = 1,
) -> tuple[HomeostaseState, list[float]]:
    """Simula o ciclo completo: lote enviesado -> checkpoint -> reponderação
    composta com os pesos anteriores -> próximo lote (agora corrigido).

    Devolve o estado final e a lista de divergências suavizadas por checkpoint.
    """
    rng = np.random.default_rng(rng_seed)
    state = HomeostaseState(contrato)
    pesos = np.ones(len(pool))
    divergencias: list[float] = []

    for i in range(n_checkpoints):
        if i == 0:
            # 1º lote: enviesado deliberadamente (simula deriva já instalada)
            bias_w = norm.pdf(pool["rendimento"], loc=vies_media, scale=vies_desvio)
            probs = bias_w / bias_w.sum()
        else:
            probs = pesos / pesos.sum()
        idx = rng.choice(len(pool), size=batch_size, replace=True, p=probs)
        lote = pool.iloc[idx].reset_index(drop=True)

        saida = run_controlador_homeostasia(lote, contrato, pool, state)
        pesos = pesos * np.array(saida.pesos_multiplicador)
        divergencias.append(saida.relatorio.divergencias_colunas[0].divergencia_suavizada)

    return state, divergencias


# --------------------------------------------------------------------------
# gaussian_kl / normalized_divergence_score
# --------------------------------------------------------------------------


def test_gaussian_kl_e_zero_quando_distribuicoes_identicas() -> None:
    assert gaussian_kl(5000, 1500, 5000, 1500) == pytest.approx(0.0, abs=1e-9)


def test_gaussian_kl_positivo_quando_distribuicoes_diferem() -> None:
    assert gaussian_kl(3000, 1500, 5000, 1500) > 0


def test_normalized_divergence_score_e_monotona_e_limitada() -> None:
    baixo = normalized_divergence_score(0.1)
    alto = normalized_divergence_score(2.0)
    assert 0 <= baixo < alto < 1


# --------------------------------------------------------------------------
# Entropia de Shannon
# --------------------------------------------------------------------------


def test_shannon_entropy_normalizada_maxima_para_uniforme() -> None:
    assert shannon_entropy_normalized({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(1.0)


def test_shannon_entropy_normalizada_minima_para_categoria_unica() -> None:
    assert shannon_entropy_normalized({"a": 1.0, "b": 0.0, "c": 0.0}) == pytest.approx(0.0, abs=1e-9)


def test_find_underrepresented_categories() -> None:
    proporcoes = {"baixo": 0.6, "medio": 0.35, "alto": 0.05}
    alvo = {"baixo": 0.5, "medio": 0.3, "alto": 0.2}
    sub = find_underrepresented_categories(proporcoes, alvo, ratio=0.5)
    assert sub == ["alto"]  # 0.05 < 0.5*0.2=0.10


# --------------------------------------------------------------------------
# Validações de input
# --------------------------------------------------------------------------


def test_run_controlador_homeostasia_lote_vazio_levanta_erro(contrato_rendimento) -> None:
    state = HomeostaseState(contrato_rendimento)
    pool = _pool_gaussiano(contrato_rendimento, 100, 1)
    with pytest.raises(HomeostaseValidationError):
        run_controlador_homeostasia(pd.DataFrame(columns=["rendimento"]), contrato_rendimento, pool, state)


def test_run_controlador_homeostasia_sem_deriva_estado_normal(contrato_rendimento) -> None:
    pool = _pool_gaussiano(contrato_rendimento, 5000, 2)
    state = HomeostaseState(contrato_rendimento)
    lote = pool.sample(1000, random_state=3).reset_index(drop=True)
    saida = run_controlador_homeostasia(lote, contrato_rendimento, pool, state)
    assert saida.relatorio.estado_global == HealthStatus.NORMAL
    assert saida.pausa_recomendada is False
    assert saida.acoes == []


# --------------------------------------------------------------------------
# Pilar 1 / Teste 1 — corrige deriva distribucional para < 2% em < 50.000 linhas
# --------------------------------------------------------------------------


def test_corrige_deriva_distribucional_abaixo_de_2_porcento_em_menos_de_50000_linhas(
    contrato_rendimento,
) -> None:
    pool = _pool_gaussiano(contrato_rendimento, 8000, 0)

    state, divergencias = _run_correction_loop(
        contrato_rendimento, pool, vies_media=4000, vies_desvio=1300, n_checkpoints=11, rng_seed=1
    )

    divergencia_inicial = divergencias[0]
    assert divergencia_inicial > 0.05  # deriva inicial significativa (ordem do "10%" do enunciado)

    linhas_adicionais_ate_50k = 10 * 5000  # 10 checkpoints após o 1º lote enviesado
    divergencia_final = divergencias[10]  # ao fim de exatamente 50.000 linhas adicionais
    assert divergencia_final < 0.02
    assert linhas_adicionais_ate_50k <= 50_000


def test_pausa_recomendada_quando_divergencia_critica(contrato_rendimento) -> None:
    pool = _pool_gaussiano(contrato_rendimento, 8000, 0)
    rng = np.random.default_rng(5)
    bias_w = norm.pdf(pool["rendimento"], loc=4000, scale=1300)
    probs = bias_w / bias_w.sum()
    idx = rng.choice(len(pool), size=5000, replace=True, p=probs)
    lote = pool.iloc[idx].reset_index(drop=True)

    state = HomeostaseState(contrato_rendimento)
    saida = run_controlador_homeostasia(lote, contrato_rendimento, pool, state)

    assert saida.relatorio.estado_global == HealthStatus.CRITICO
    assert saida.pausa_recomendada is True
    assert any("CRÍTICA" in w for w in saida.warnings)


# --------------------------------------------------------------------------
# Pilar 1 / Teste 2 — não reage exageradamente a ruído amostral
# --------------------------------------------------------------------------


def test_nao_reage_a_flutuacoes_amostrais_ruido(contrato_rendimento) -> None:
    """Lotes pequenos amostrados EXATAMENTE da distribuição do Contrato
    (sem deriva real) produzem, por acaso, picos instantâneos de
    divergência — mas a versão suavizada (EMA) nunca deve escalar para
    ATENÇÃO/CRÍTICO por causa desse ruído.
    """
    pool = _pool_gaussiano(contrato_rendimento, 20000, 10)
    state = HomeostaseState(contrato_rendimento)
    rng = np.random.default_rng(11)

    picos_instantaneos = []
    estados_suavizados = []
    for _ in range(15):
        lote = pool.sample(30, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)
        saida = run_controlador_homeostasia(lote, contrato_rendimento, pool, state)
        rep = saida.relatorio.divergencias_colunas[0]
        picos_instantaneos.append(rep.divergencia_normalizada)
        estados_suavizados.append(rep.estado)

    # sanidade: o ruído amostral (lotes pequenos) de facto produz picos
    assert max(picos_instantaneos) > 0.05
    # mas a versão suavizada nunca escala para ATENÇÃO/CRÍTICO
    assert all(estado == HealthStatus.NORMAL for estado in estados_suavizados)


def test_deriva_persistente_ainda_e_detectada_apesar_da_suavizacao(contrato_rendimento) -> None:
    """Ao contrário do ruído (Teste 2), uma deriva real e persistente
    deve ser detectada mesmo com o filtro de média móvel suavizada.
    """
    pool = _pool_gaussiano(contrato_rendimento, 8000, 0)
    state, divergencias = _run_correction_loop(
        contrato_rendimento, pool, vies_media=3000, vies_desvio=1200, n_checkpoints=1, rng_seed=2
    )
    assert divergencias[0] > 0.05


# --------------------------------------------------------------------------
# Pilar 1 / Teste 3 — mantém diversidade (entropia) ao corrigir a distribuição
# --------------------------------------------------------------------------


def test_mantem_entropia_alta_enquanto_corrige_distribuicao_numerica() -> None:
    """Cenário combinado: deriva numérica (rendimento) + défice de
    diversidade categórica (uma categoria rara sub-representada) na
    MESMA sequência de lotes — a correção de um não pode sacrificar o outro.
    """
    column_stats = {
        "rendimento": _continuous(5000, 1500),
        "risco": CategoricalStatistics(frequencias={"baixo": 0.5, "medio": 0.3, "alto": 0.2}),
    }
    contrato = _make_contract(column_stats)

    rng = np.random.default_rng(0)
    n = 8000
    rendimento_pool = rng.normal(5000, 1500, n)
    rendimento_pool = rendimento_pool[rendimento_pool > 0]
    categorias = rng.choice(["baixo", "medio", "alto"], size=len(rendimento_pool), p=[0.5, 0.3, 0.2])
    pool = pd.DataFrame({"rendimento": rendimento_pool, "risco": categorias}).reset_index(drop=True)

    state = HomeostaseState(contrato)
    pesos = np.ones(len(pool))
    rng2 = np.random.default_rng(3)

    # 1º lote: enviesado tanto na média do rendimento como quase sem
    # representação da categoria "alto" (défice de diversidade)
    bias_w = norm.pdf(pool["rendimento"], loc=4000, scale=1300)
    bias_w = bias_w * np.where(pool["risco"] == "alto", 0.05, 1.0)
    probs = bias_w / bias_w.sum()
    idx = rng2.choice(len(pool), size=5000, replace=True, p=probs)
    lote0 = pool.iloc[idx].reset_index(drop=True)
    saida = run_controlador_homeostasia(lote0, contrato, pool, state)
    entropia_inicial = saida.relatorio.entropias[0].entropia_normalizada
    pesos = pesos * np.array(saida.pesos_multiplicador)

    for _ in range(10):
        p = pesos / pesos.sum()
        idx = rng2.choice(len(pool), size=5000, replace=True, p=p)
        lote = pool.iloc[idx].reset_index(drop=True)
        saida = run_controlador_homeostasia(lote, contrato, pool, state)
        pesos = pesos * np.array(saida.pesos_multiplicador)

    divergencia_final = saida.relatorio.divergencias_colunas[0].divergencia_suavizada
    entropia_final = saida.relatorio.entropias[0].entropia_normalizada

    assert entropia_inicial < 0.7  # défice de diversidade inicial confirmado
    assert divergencia_final < 0.05  # a deriva numérica foi corrigida
    assert entropia_final >= 0.7  # ...sem sacrificar a diversidade categórica
    assert entropia_final > entropia_inicial


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end
# --------------------------------------------------------------------------


def test_run_controlador_homeostasia_end_to_end(contrato_rendimento) -> None:
    """Testa o módulo como um todo: lote -> atualização de estado ->
    relatório -> ações -> multiplicadores de peso alinhados com a memória.
    """
    pool = _pool_gaussiano(contrato_rendimento, 5000, 7)
    state = HomeostaseState(contrato_rendimento)
    rng = np.random.default_rng(8)
    bias_w = norm.pdf(pool["rendimento"], loc=4000, scale=1300)
    probs = bias_w / bias_w.sum()
    idx = rng.choice(len(pool), size=3000, replace=True, p=probs)
    lote = pool.iloc[idx].reset_index(drop=True)

    saida = run_controlador_homeostasia(lote, contrato_rendimento, pool, state)

    assert saida.relatorio.n_linhas_acumuladas == 3000
    assert len(saida.pesos_multiplicador) == len(pool)
    assert all(p > 0 for p in saida.pesos_multiplicador)
    assert len(saida.acoes) >= 1
    assert saida.acoes[0].coluna_alvo == "rendimento"
