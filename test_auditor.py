"""Testes do módulo Auditor de Utilidade, Privacidade e Valor Prático."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synap.auditor import (
    AuditorValidationError,
    InsufficientHoldoutError,
    assess_privacy,
    assess_utility,
    dcr_assessment,
    exact_match_assessment,
    membership_inference_risk,
    nndr_assessment,
    run_auditor,
    select_target_columns,
)
from synap.neocortex.models import (
    CategoricalStatistics,
    ContinuousStatistics,
    ContractSource,
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


def _make_contract(column_stats: dict) -> StatisticalContract:
    return StatisticalContract(
        column_stats=column_stats,
        correlations=[],
        tolerancia_kl=0.05,
        source=ContractSource.REAL_CSV,
        is_synthetic_pure=False,
        sample_size=1000,
        rules=[],
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
    rng = np.random.default_rng(rng_seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = coef * x1 + rng.normal(0, noise, n)
    p = _sigmoid(logit)
    y = np.where(rng.uniform(0, 1, n) < p, "1", "0")
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_decorrelated_dataset(n: int, rng_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = rng.choice(["0", "1"], size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_memorizing_synthetic(real_holdout: pd.DataFrame, n_total: int, frac_copiado: float, rng_seed: int) -> pd.DataFrame:
    """Constrói um "sintético" que copia uma fração das linhas do holdout real (memorização)."""
    n_copiado = int(n_total * frac_copiado)
    copiadas = real_holdout.sample(n_copiado, random_state=rng_seed, replace=True).reset_index(drop=True)
    resto = _make_decorrelated_dataset(n_total - n_copiado, rng_seed + 1)
    return pd.concat([copiadas, resto], ignore_index=True)


# --------------------------------------------------------------------------
# select_target_columns
# --------------------------------------------------------------------------


def test_select_target_columns_usa_lista_fornecida(contrato_binario) -> None:
    assert select_target_columns(contrato_binario, ["x1"]) == ["x1"]


def test_select_target_columns_rejeita_coluna_inexistente(contrato_binario) -> None:
    with pytest.raises(AuditorValidationError):
        select_target_columns(contrato_binario, ["inexistente"])


def test_select_target_columns_prioriza_categoricas(contrato_binario) -> None:
    alvos = select_target_columns(contrato_binario)
    assert alvos[0] == "y"


# --------------------------------------------------------------------------
# Pilar 1 — Utilidade estatística (TSTR/TRTR)
# --------------------------------------------------------------------------


def test_assess_utility_alta_para_sintetico_fiel(contrato_binario) -> None:
    sintetico = _make_binary_dataset(1000, coef=2.0, noise=0.6, rng_seed=1)
    real = _make_binary_dataset(1000, coef=2.0, noise=0.6, rng_seed=2)
    avaliacao = assess_utility(sintetico, real, contrato_binario, target_columns=["y"], rng_seed=0)
    assert avaliacao.aprovado is True
    assert avaliacao.retencao_media >= 0.85


def test_assess_utility_baixa_para_sintetico_com_relacao_invertida(contrato_binario) -> None:
    # Nota: usar y puramente aleatório (independente de x1/x2) produz um
    # AUC instável (0.1 a 0.9 consoante a seed) porque a AUC é uma
    # métrica de ranking — mesmo um coeficiente espúrio ínfimo, mas com
    # sinal consistente, desloca fortemente o ranking. Uma relação
    # INVERTIDA (mesma força, sinal oposto) dá um AUC baixo e estável.
    sintetico = _make_binary_dataset(1000, coef=-2.5, noise=0.4, rng_seed=1)
    real = _make_binary_dataset(1000, coef=2.5, noise=0.4, rng_seed=2)
    avaliacao = assess_utility(sintetico, real, contrato_binario, target_columns=["y"], rng_seed=0)
    assert avaliacao.aprovado is False
    assert avaliacao.retencao_media < 0.85


# --------------------------------------------------------------------------
# Pilar 2 — DCR / NNDR
# --------------------------------------------------------------------------


def test_dcr_razao_proxima_de_1_para_sintetico_independente(contrato_binario) -> None:
    real = _make_binary_dataset(600, coef=2.0, noise=0.6, rng_seed=1)
    sintetico = _make_binary_dataset(600, coef=2.0, noise=0.6, rng_seed=2)
    avaliacao = dcr_assessment(sintetico, real, contrato_binario, rng_seed=0)
    assert avaliacao.razao > 0.5
    # alguma proximidade natural é normal (não é memorização); só uma
    # pequena fração das linhas deve cair no limiar de suspeita
    assert avaliacao.linhas_suspeitas < 0.05 * len(sintetico)


def test_dcr_razao_baixa_para_sintetico_memorizado(contrato_binario) -> None:
    real = _make_binary_dataset(600, coef=2.0, noise=0.6, rng_seed=1)
    sintetico_memorizado = _make_memorizing_synthetic(real, n_total=600, frac_copiado=0.3, rng_seed=5)
    avaliacao = dcr_assessment(sintetico_memorizado, real, contrato_binario, rng_seed=0)
    # com apenas 30% de linhas copiadas, a MEDIANA global pode não descer
    # abaixo do limiar (a maioria das linhas continua independente) — o
    # sinal correto de memorização PARCIAL é a contagem de linhas
    # suspeitas, que deve aproximar-se da fração efetivamente copiada
    assert avaliacao.linhas_suspeitas >= 0.2 * len(sintetico_memorizado)
    assert avaliacao.percentil_5_sintetico_real < avaliacao.dcr_real_real_mediana * 0.1


def test_nndr_executa_e_devolve_valores_positivos(contrato_binario) -> None:
    real = _make_binary_dataset(400, coef=2.0, noise=0.6, rng_seed=1)
    sintetico = _make_binary_dataset(400, coef=2.0, noise=0.6, rng_seed=2)
    avaliacao = nndr_assessment(sintetico, real, contrato_binario, rng_seed=0)
    assert avaliacao.nndr_sintetico_mediana > 0
    assert avaliacao.nndr_real_mediana > 0


# --------------------------------------------------------------------------
# Pilar 2 — Correspondência exacta
# --------------------------------------------------------------------------


def test_exact_match_zero_para_dados_independentes(contrato_binario) -> None:
    real = _make_binary_dataset(300, coef=2.0, noise=0.6, rng_seed=1)
    sintetico = _make_binary_dataset(300, coef=2.0, noise=0.6, rng_seed=2)
    avaliacao = exact_match_assessment(sintetico, real, contrato_binario)
    assert avaliacao.n_correspondencias_exactas == 0


def test_exact_match_deteta_copias_literais(contrato_binario) -> None:
    real = _make_binary_dataset(300, coef=2.0, noise=0.6, rng_seed=1)
    sintetico = pd.concat([real.iloc[:20], _make_decorrelated_dataset(100, rng_seed=9)], ignore_index=True)
    avaliacao = exact_match_assessment(sintetico, real, contrato_binario)
    assert avaliacao.n_correspondencias_exactas == 20


# --------------------------------------------------------------------------
# Pilar 2 — Ataque de Inferência de Pertença (MIA)
# --------------------------------------------------------------------------


def test_mia_nao_avaliavel_sem_training_reference(contrato_binario) -> None:
    real = _make_binary_dataset(300, coef=2.0, noise=0.6, rng_seed=1)
    sintetico = _make_binary_dataset(300, coef=2.0, noise=0.6, rng_seed=2)
    avaliacao = membership_inference_risk(sintetico, real, None, contrato_binario, rng_seed=0)
    assert avaliacao.avaliavel is False
    assert avaliacao.auc is None


def test_mia_seguro_quando_nao_ha_memorizacao(contrato_binario) -> None:
    real = _make_binary_dataset(400, coef=2.0, noise=0.6, rng_seed=1)
    training_reference = _make_binary_dataset(400, coef=2.0, noise=0.6, rng_seed=3)  # independente
    sintetico = _make_binary_dataset(400, coef=2.0, noise=0.6, rng_seed=2)  # também independente

    avaliacao = membership_inference_risk(sintetico, real, training_reference, contrato_binario, rng_seed=0)
    assert avaliacao.avaliavel is True
    assert avaliacao.seguro is True


def test_mia_deteta_memorizacao_de_training_reference(contrato_binario) -> None:
    real = _make_binary_dataset(400, coef=2.0, noise=0.6, rng_seed=1)
    training_reference = _make_binary_dataset(400, coef=2.0, noise=0.6, rng_seed=3)
    # o sintético memoriza (copia) as linhas de training_reference
    sintetico = _make_memorizing_synthetic(training_reference, n_total=400, frac_copiado=0.5, rng_seed=7)

    avaliacao = membership_inference_risk(sintetico, real, training_reference, contrato_binario, rng_seed=0)
    assert avaliacao.avaliavel is True
    assert avaliacao.seguro is False
    assert avaliacao.auc > 0.6


# --------------------------------------------------------------------------
# assess_privacy (agregado)
# --------------------------------------------------------------------------


def test_assess_privacy_nao_avaliavel_sem_holdout(contrato_binario) -> None:
    sintetico = _make_binary_dataset(100, coef=2.0, noise=0.6, rng_seed=1)
    avaliacao = assess_privacy(sintetico, None, contrato_binario)
    assert avaliacao.avaliavel is False
    assert avaliacao.aprovado is False


def test_assess_privacy_aprova_dataset_seguro(contrato_binario) -> None:
    real = _make_binary_dataset(600, coef=2.0, noise=0.6, rng_seed=1)
    sintetico = _make_binary_dataset(600, coef=2.0, noise=0.6, rng_seed=2)
    avaliacao = assess_privacy(sintetico, real, contrato_binario, rng_seed=0)
    assert avaliacao.aprovado is True
    assert avaliacao.motivo is None


def test_assess_privacy_bloqueia_dataset_memorizado(contrato_binario) -> None:
    real = _make_binary_dataset(600, coef=2.0, noise=0.6, rng_seed=1)
    sintetico_memorizado = _make_memorizing_synthetic(real, n_total=600, frac_copiado=0.3, rng_seed=5)
    avaliacao = assess_privacy(sintetico_memorizado, real, contrato_binario, rng_seed=0)
    assert avaliacao.aprovado is False
    assert avaliacao.motivo is not None


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end, incluindo o veto de privacidade
# --------------------------------------------------------------------------


def test_run_auditor_sem_holdout_levanta_erro(contrato_binario) -> None:
    sintetico = _make_binary_dataset(100, coef=2.0, noise=0.6, rng_seed=1)
    with pytest.raises(InsufficientHoldoutError):
        run_auditor(sintetico, contrato_binario, real_holdout=None)


def test_run_auditor_aprova_dataset_util_e_privado(tmp_path, contrato_binario) -> None:
    """Testa o módulo como um todo: utilidade + privacidade + relatório."""
    real = _make_binary_dataset(800, coef=2.0, noise=0.6, rng_seed=1)
    sintetico = _make_binary_dataset(800, coef=2.0, noise=0.6, rng_seed=2)

    resultado = run_auditor(
        sintetico, contrato_binario, real_holdout=real, output_dir=tmp_path, rng_seed=0
    )

    assert resultado.aprovado_global is True
    assert resultado.diagnostico is None
    assert Path(resultado.report_path).exists()
    assert Path(resultado.report_path).stat().st_size > 0


def test_run_auditor_privacidade_veta_apesar_de_utilidade_alta(tmp_path, contrato_binario) -> None:
    """O caso central do Módulo 8: utilidade alta NÃO basta se houver memorização."""
    real = _make_binary_dataset(800, coef=2.0, noise=0.6, rng_seed=1)
    # sintético útil (mesma relação x1->y) MAS parcialmente memorizado
    sintetico_util_mas_memorizado = pd.concat(
        [
            _make_binary_dataset(560, coef=2.0, noise=0.6, rng_seed=2),
            real.sample(240, random_state=42, replace=True).reset_index(drop=True),
        ],
        ignore_index=True,
    )

    resultado = run_auditor(
        sintetico_util_mas_memorizado, contrato_binario, real_holdout=real, output_dir=tmp_path, rng_seed=0
    )

    assert resultado.utilidade.aprovado is True  # a utilidade, isolada, passaria
    assert resultado.privacidade.aprovado is False  # mas a privacidade veta
    assert resultado.aprovado_global is False
    assert "privacidade" in resultado.diagnostico.lower()


def test_run_auditor_com_training_reference_avalia_mia(tmp_path, contrato_binario) -> None:
    real = _make_binary_dataset(500, coef=2.0, noise=0.6, rng_seed=1)
    training_reference = _make_binary_dataset(500, coef=2.0, noise=0.6, rng_seed=3)
    sintetico = _make_binary_dataset(500, coef=2.0, noise=0.6, rng_seed=2)

    resultado = run_auditor(
        sintetico,
        contrato_binario,
        real_holdout=real,
        training_reference=training_reference,
        output_dir=tmp_path,
        rng_seed=0,
    )

    assert resultado.privacidade.mia.avaliavel is True
    assert resultado.warnings == []  # training_reference fornecido, sem aviso de MIA não avaliado
