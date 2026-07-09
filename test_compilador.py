"""Testes do módulo Neocórtex Motor, Validador por Proxy e Compilador de Confiança."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from scipy import stats as scipy_stats

from synap.compilador import (
    CompiladorValidationError,
    compute_fidelity_index,
    run_compilador,
    run_final_asp_validation,
    run_twin_model_validation,
    select_target_column,
    train_twin_model,
)
from synap.compilador.models import TargetType, TwinModelMetrics
from synap.neocortex.models import (
    BusinessRule,
    CategoricalStatistics,
    ContinuousStatistics,
    ContractSource,
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


def _make_contract(column_stats: dict, rules: list[BusinessRule] | None = None) -> StatisticalContract:
    return StatisticalContract(
        column_stats=column_stats,
        correlations=[],
        tolerancia_kl=0.05,
        source=ContractSource.REAL_CSV,
        is_synthetic_pure=False,
        sample_size=1000,
        rules=rules or [],
        warnings=[],
    )


@pytest.fixture
def contrato_binario() -> StatisticalContract:
    column_stats = {
        "x1": _continuous(0, 1),
        "x2": _continuous(0, 1),
        "y": CategoricalStatistics(frequencias={"0": 0.5, "1": 0.5}),
    }
    return _make_contract(column_stats)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _make_binary_dataset(n: int, coef: float, noise: float, rng_seed: int) -> pd.DataFrame:
    """Gera um dataset onde `y` depende logisticamente de `x1` com coeficiente `coef`."""
    rng = np.random.default_rng(rng_seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = coef * x1 + rng.normal(0, noise, n)
    p = _sigmoid(logit)
    y = np.where(rng.uniform(0, 1, n) < p, "1", "0")
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_decorrelated_dataset(n: int, rng_seed: int) -> pd.DataFrame:
    """Gera um dataset onde `y` é INDEPENDENTE de `x1`/`x2` (relação real quebrada)."""
    rng = np.random.default_rng(rng_seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = rng.choice(["0", "1"], size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# --------------------------------------------------------------------------
# select_target_column
# --------------------------------------------------------------------------


def test_select_target_column_usa_coluna_fornecida(contrato_binario) -> None:
    assert select_target_column(contrato_binario, "x2") == "x2"


def test_select_target_column_rejeita_coluna_inexistente(contrato_binario) -> None:
    with pytest.raises(CompiladorValidationError):
        select_target_column(contrato_binario, "inexistente")


def test_select_target_column_escolhe_maior_variancia() -> None:
    column_stats = {"a": _continuous(0, 1), "b": _continuous(0, 10), "c": _continuous(0, 5)}
    contrato = _make_contract(column_stats)
    assert select_target_column(contrato) == "b"


# --------------------------------------------------------------------------
# train_twin_model / validação de input
# --------------------------------------------------------------------------


def test_train_twin_model_dataset_pequeno_levanta_erro(contrato_binario) -> None:
    df = _make_binary_dataset(20, coef=2.0, noise=0.5, rng_seed=1)
    with pytest.raises(CompiladorValidationError):
        train_twin_model(df, contrato_binario, "y")


def test_train_twin_model_classificacao_devolve_auc(contrato_binario) -> None:
    df = _make_binary_dataset(500, coef=2.0, noise=0.5, rng_seed=1)
    metrica = train_twin_model(df, contrato_binario, "y", rng_seed=0)
    assert metrica.tipo_alvo == TargetType.CATEGORICO
    assert metrica.metrica_nome == "AUC"
    assert 0.5 < metrica.valor <= 1.0


def test_train_twin_model_regressao_devolve_rmse() -> None:
    column_stats = {"x1": _continuous(0, 1), "alvo": _continuous(0, 5)}
    contrato = _make_contract(column_stats)
    rng = np.random.default_rng(2)
    x1 = rng.normal(0, 1, 500)
    alvo = 3 * x1 + rng.normal(0, 1, 500)
    df = pd.DataFrame({"x1": x1, "alvo": alvo})
    metrica = train_twin_model(df, contrato, "alvo", rng_seed=0)
    assert metrica.tipo_alvo == TargetType.NUMERICO
    assert metrica.metrica_nome == "RMSE"
    assert metrica.valor > 0


# --------------------------------------------------------------------------
# compute_fidelity_index
# --------------------------------------------------------------------------


def test_compute_fidelity_index_identico_e_um() -> None:
    m1 = TwinModelMetrics(tipo_alvo=TargetType.CATEGORICO, metrica_nome="AUC", valor=0.85, n_treino=1, n_teste=1)
    m2 = TwinModelMetrics(tipo_alvo=TargetType.CATEGORICO, metrica_nome="AUC", valor=0.85, n_treino=1, n_teste=1)
    assert compute_fidelity_index(m1, m2) == pytest.approx(1.0)


def test_compute_fidelity_index_exemplo_do_enunciado() -> None:
    sintetica = TwinModelMetrics(tipo_alvo=TargetType.CATEGORICO, metrica_nome="AUC", valor=0.82, n_treino=1, n_teste=1)
    real = TwinModelMetrics(tipo_alvo=TargetType.CATEGORICO, metrica_nome="AUC", valor=0.85, n_treino=1, n_teste=1)
    assert compute_fidelity_index(sintetica, real) == pytest.approx(0.9647, abs=1e-3)


# --------------------------------------------------------------------------
# run_final_asp_validation
# --------------------------------------------------------------------------


def test_run_final_asp_validation_aprova_dataset_valido(contrato_binario) -> None:
    df = _make_binary_dataset(200, coef=1.5, noise=0.5, rng_seed=3)
    resultado = run_final_asp_validation(df, contrato_binario)
    assert resultado.aprovado is True
    assert resultado.rejeitadas == 0


def test_run_final_asp_validation_rejeita_dataset_invalido() -> None:
    column_stats = {"rendimento": _continuous(3000, 500)}
    regra = BusinessRule(coluna="rendimento", operador=RuleOperator.GT, valor=0, texto_original="rendimento > 0")
    contrato = _make_contract(column_stats, rules=[regra])
    df = pd.DataFrame({"rendimento": [1000, -50, 2000]})
    resultado = run_final_asp_validation(df, contrato)
    assert resultado.aprovado is False
    assert resultado.rejeitadas == 1


# --------------------------------------------------------------------------
# Pilar 4 / Teste 1 — Modelo Gémeo estatisticamente indistinguível (p > 0.05)
# --------------------------------------------------------------------------


def test_modelo_gemeo_indistinguivel_para_datasets_de_alta_fidelidade(contrato_binario) -> None:
    """Sintético e real gerados pelo MESMO processo gerador — a
    distribuição de AUCs ao longo de várias amostras independentes
    (não apenas splits diferentes da mesma amostra) não deve diferir
    significativamente.
    """
    aucs_sinteticos = [
        train_twin_model(
            _make_binary_dataset(800, coef=1.5, noise=0.8, rng_seed=5000 + i), contrato_binario, "y", rng_seed=0
        ).valor
        for i in range(12)
    ]
    aucs_reais = [
        train_twin_model(
            _make_binary_dataset(800, coef=1.5, noise=0.8, rng_seed=6000 + i), contrato_binario, "y", rng_seed=0
        ).valor
        for i in range(12)
    ]

    _, p_valor = scipy_stats.ttest_ind(aucs_sinteticos, aucs_reais)
    assert p_valor > 0.05


def test_run_twin_model_validation_indice_alto_para_alta_fidelidade(contrato_binario) -> None:
    sintetico = _make_binary_dataset(1500, coef=2.0, noise=0.6, rng_seed=10)
    real = _make_binary_dataset(1500, coef=2.0, noise=0.6, rng_seed=20)
    avaliacao = run_twin_model_validation(sintetico, contrato_binario, real_holdout=real, rng_seed=0)
    assert avaliacao.indice_fidelidade >= 0.90


# --------------------------------------------------------------------------
# Pilar 4 / Teste 2 — bloqueia dataset com relação real quebrada (Índice < 0.80)
# --------------------------------------------------------------------------


def test_bloqueia_dataset_com_relacao_quebrada_indice_abaixo_de_080(contrato_binario) -> None:
    # "real": relação forte x1->y (AUC alto); "sintético": relação destruída (AUC ~0.5)
    real = _make_binary_dataset(1500, coef=2.5, noise=0.4, rng_seed=1)
    sintetico_quebrado = _make_decorrelated_dataset(1500, rng_seed=2)

    avaliacao = run_twin_model_validation(
        sintetico_quebrado, contrato_binario, real_holdout=real, rng_seed=0
    )
    assert avaliacao.indice_fidelidade < 0.80


def test_run_compilador_bloqueia_saida_quando_fidelidade_baixa(tmp_path, contrato_binario) -> None:
    real = _make_binary_dataset(1500, coef=2.5, noise=0.4, rng_seed=1)
    sintetico_quebrado = _make_decorrelated_dataset(1500, rng_seed=2)

    resultado = run_compilador(
        sintetico_quebrado,
        contrato_binario,
        real_holdout=real,
        output_dir=tmp_path,
        rng_seed=0,
    )

    assert resultado.liberado is False
    assert resultado.csv_path is None
    assert resultado.report_path is None
    assert "Motivo" in resultado.diagnostico


def test_run_compilador_forca_saida_apesar_de_fidelidade_baixa(tmp_path, contrato_binario) -> None:
    real = _make_binary_dataset(1500, coef=2.5, noise=0.4, rng_seed=1)
    sintetico_quebrado = _make_decorrelated_dataset(1500, rng_seed=2)

    resultado = run_compilador(
        sintetico_quebrado,
        contrato_binario,
        real_holdout=real,
        output_dir=tmp_path,
        forcar_saida=True,
        rng_seed=0,
    )

    assert resultado.liberado is True
    assert resultado.csv_path is not None
    assert any("FORÇADA" in w for w in resultado.warnings)


def test_run_compilador_bloqueia_por_violacao_asp_antes_do_modelo_gemeo(tmp_path) -> None:
    column_stats = {"rendimento": _continuous(3000, 500), "y": CategoricalStatistics(frequencias={"0": 0.5, "1": 0.5})}
    regra = BusinessRule(coluna="rendimento", operador=RuleOperator.GT, valor=0, texto_original="rendimento > 0")
    contrato = _make_contract(column_stats, rules=[regra])
    df = pd.DataFrame({"rendimento": [1000, -50] * 100, "y": ["0", "1"] * 100})

    resultado = run_compilador(df, contrato, output_dir=tmp_path, forcar_saida=True)

    assert resultado.liberado is False
    assert resultado.fidelidade is None  # nem chegou a treinar o Modelo Gémeo
    assert "ASP" in resultado.diagnostico


def test_run_compilador_sem_holdout_bloqueia_por_omissao(tmp_path, contrato_binario) -> None:
    sintetico = _make_binary_dataset(500, coef=2.0, noise=0.5, rng_seed=1)
    resultado = run_compilador(sintetico, contrato_binario, output_dir=tmp_path)
    assert resultado.liberado is False
    assert resultado.fidelidade.indice_fidelidade is None


# --------------------------------------------------------------------------
# Pilar 4 / Teste 3 — relatório autoexplicativo (estrutura verificável)
# --------------------------------------------------------------------------


def test_run_compilador_libera_e_gera_relatorio_completo(tmp_path, contrato_binario) -> None:
    """Testa o módulo como um todo: ASP -> Modelo Gémeo -> Índice de
    Fidelidade -> CSV + Relatório de Confiança em PDF com múltiplas páginas.
    """
    sintetico = _make_binary_dataset(1500, coef=2.0, noise=0.6, rng_seed=10)
    real = _make_binary_dataset(1500, coef=2.0, noise=0.6, rng_seed=20)

    resultado = run_compilador(
        sintetico,
        contrato_binario,
        real_holdout=real,
        output_dir=tmp_path,
        rng_seed=0,
    )

    assert resultado.liberado is True
    assert resultado.diagnostico is None
    assert Path(resultado.csv_path).exists()
    assert Path(resultado.report_path).exists()

    csv_lido = pd.read_csv(resultado.csv_path)
    assert list(csv_lido.columns) == ["x1", "x2", "y"]
    assert len(csv_lido) == len(sintetico)

    leitor_pdf = PdfReader(resultado.report_path)
    # resumo + histogramas + correlação + comparação do gémeo = pelo menos 4 páginas
    assert len(leitor_pdf.pages) >= 4
    texto_pagina1 = leitor_pdf.pages[0].extract_text()
    assert "Índice de Fidelidade" in texto_pagina1
