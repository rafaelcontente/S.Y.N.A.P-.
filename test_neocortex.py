"""Testes do módulo Neocórtex Perceptual e Ancoragem Estatística."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from synap.neocortex import (
    ContractSource,
    InsufficientSampleError,
    NeocortexValidationError,
    build_contract,
    load_schema,
    parse_business_rules,
)
from synap.neocortex.models import SchemaDefinition
from synap.neocortex.neocortex import MIN_SAMPLE_SIZE_HARD, MIN_SAMPLE_SIZE_RECOMMENDED

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def schema_dict() -> dict:
    """Esquema mínimo com colunas numéricas e categóricas."""
    return {
        "colunas": {
            "idade": {"tipo": "numerico", "minimo": 30, "maximo": 50},
            "rendimento": {"tipo": "numerico", "minimo": 0},
            "risco": {"tipo": "categorico", "categorias": ["baixo", "medio", "alto"]},
        },
        "regras": ["rendimento > 0"],
    }


@pytest.fixture
def schema_path(tmp_path: Path, schema_dict: dict) -> Path:
    """Escreve o esquema num ficheiro YAML temporário."""
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(schema_dict), encoding="utf-8")
    return path


@pytest.fixture
def schema_json_path(tmp_path: Path, schema_dict: dict) -> Path:
    """Escreve o esquema num ficheiro JSON temporário."""
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema_dict), encoding="utf-8")
    return path


def _make_correlated_csv(path: Path, n_rows: int, n_cols: int = 10, seed: int = 42) -> None:
    """Gera um CSV sintético com `n_cols` colunas numéricas correlacionadas."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=40, scale=10, size=n_rows)
    data = {"idade": base, "rendimento": base * 120 + rng.normal(0, 500, n_rows)}
    for i in range(n_cols - 2):
        data[f"coluna_{i}"] = rng.normal(loc=0, scale=1, size=n_rows)
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.fixture
def large_schema_path(tmp_path: Path) -> Path:
    """Esquema com 10 colunas numéricas, coerente com `_make_correlated_csv`."""
    colunas = {"idade": {"tipo": "numerico"}, "rendimento": {"tipo": "numerico"}}
    for i in range(8):
        colunas[f"coluna_{i}"] = {"tipo": "numerico"}
    path = tmp_path / "schema_grande.yaml"
    path.write_text(yaml.safe_dump({"colunas": colunas, "regras": []}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# load_schema
# --------------------------------------------------------------------------


def test_load_schema_valid_yaml_retorna_schema_definition(schema_path: Path) -> None:
    schema = load_schema(schema_path)
    assert isinstance(schema, SchemaDefinition)
    assert set(schema.colunas) == {"idade", "rendimento", "risco"}


def test_load_schema_valid_json_retorna_schema_definition(schema_json_path: Path) -> None:
    schema = load_schema(schema_json_path)
    assert isinstance(schema, SchemaDefinition)


def test_load_schema_ficheiro_inexistente_levanta_erro(tmp_path: Path) -> None:
    with pytest.raises(NeocortexValidationError, match="não encontrado"):
        load_schema(tmp_path / "nao_existe.yaml")


def test_load_schema_extensao_nao_suportada_levanta_erro(tmp_path: Path) -> None:
    path = tmp_path / "schema.txt"
    path.write_text("idade: numerico", encoding="utf-8")
    with pytest.raises(NeocortexValidationError, match="extensão"):
        load_schema(path)


def test_load_schema_yaml_malformado_levanta_erro(tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"
    path.write_text("colunas: [invalido: : :", encoding="utf-8")
    with pytest.raises(NeocortexValidationError):
        load_schema(path)


def test_load_schema_dominio_invertido_levanta_erro(tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"
    path.write_text(
        yaml.safe_dump({"colunas": {"idade": {"tipo": "numerico", "minimo": 50, "maximo": 30}}}),
        encoding="utf-8",
    )
    with pytest.raises(NeocortexValidationError):
        load_schema(path)


# --------------------------------------------------------------------------
# parse_business_rules
# --------------------------------------------------------------------------


def test_parse_business_rules_regra_valida(schema_path: Path) -> None:
    schema = load_schema(schema_path)
    rules = parse_business_rules(["rendimento > 0"], schema)
    assert len(rules) == 1
    assert rules[0].coluna == "rendimento"
    assert rules[0].valor == 0.0


def test_parse_business_rules_formato_invalido_levanta_erro(schema_path: Path) -> None:
    schema = load_schema(schema_path)
    with pytest.raises(NeocortexValidationError, match="malformada"):
        parse_business_rules(["rendimento maior que 0"], schema)


def test_parse_business_rules_coluna_inexistente_levanta_erro(schema_path: Path) -> None:
    schema = load_schema(schema_path)
    with pytest.raises(NeocortexValidationError, match="inexistente"):
        parse_business_rules(["salario > 0"], schema)


# --------------------------------------------------------------------------
# Pilar 1 / Teste 1 — correlação com 10 colunas / 500 linhas + rejeição < 50
# --------------------------------------------------------------------------


def test_build_contract_calcula_correlacao_10_colunas_500_linhas(
    tmp_path: Path, large_schema_path: Path
) -> None:
    csv_path = tmp_path / "dados_grandes.csv"
    _make_correlated_csv(csv_path, n_rows=500, n_cols=10)

    contract = build_contract(schema_path=large_schema_path, anchor_csv=csv_path)

    assert contract.source == ContractSource.REAL_CSV
    assert contract.sample_size == 500
    assert len(contract.column_stats) == 10
    # 10 colunas numéricas -> C(10,2) = 45 pares de correlação
    assert len(contract.correlations) == 45
    par_idade_rendimento = next(
        p
        for p in contract.correlations
        if {p.coluna_a, p.coluna_b} == {"idade", "rendimento"}
    )
    # idade e rendimento foram construídas fortemente correlacionadas
    assert par_idade_rendimento.valor > 0.8


def test_build_contract_rejeita_amostra_abaixo_de_50_linhas(
    tmp_path: Path, schema_path: Path
) -> None:
    csv_path = tmp_path / "dados_pequenos.csv"
    pd.DataFrame(
        {
            "idade": np.linspace(30, 50, 30),
            "rendimento": np.linspace(1000, 5000, 30),
            "risco": ["baixo"] * 30,
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(InsufficientSampleError, match="parâmetros manuais"):
        build_contract(schema_path=schema_path, anchor_csv=csv_path)


def test_build_contract_amostra_entre_50_e_100_gera_aviso(
    tmp_path: Path, schema_path: Path
) -> None:
    csv_path = tmp_path / "dados_medios.csv"
    n = MIN_SAMPLE_SIZE_HARD + 10
    assert n < MIN_SAMPLE_SIZE_RECOMMENDED
    pd.DataFrame(
        {
            "idade": np.linspace(30, 50, n),
            "rendimento": np.linspace(1000, 5000, n),
            "risco": ["baixo"] * n,
        }
    ).to_csv(csv_path, index=False)

    contract = build_contract(schema_path=schema_path, anchor_csv=csv_path)

    assert contract.sample_size == n
    assert any("amostra pequena" in w for w in contract.warnings)


def test_build_contract_amostra_acima_de_100_nao_gera_aviso_de_tamanho(
    tmp_path: Path, schema_path: Path
) -> None:
    csv_path = tmp_path / "dados_suficientes.csv"
    n = MIN_SAMPLE_SIZE_RECOMMENDED + 50
    pd.DataFrame(
        {
            "idade": np.linspace(30, 50, n),
            "rendimento": np.linspace(1000, 5000, n),
            "risco": ["baixo", "medio", "alto"] * (n // 3) + ["baixo"] * (n % 3),
        }
    ).to_csv(csv_path, index=False)

    contract = build_contract(schema_path=schema_path, anchor_csv=csv_path)

    assert not any("amostra pequena" in w for w in contract.warnings)


def test_build_contract_correlacao_omite_coluna_constante(
    tmp_path: Path, schema_path: Path
) -> None:
    csv_path = tmp_path / "dados_constantes.csv"
    n = 100
    pd.DataFrame(
        {
            "idade": np.linspace(30, 50, n),
            "rendimento": [5000.0] * n,  # variância nula
            "risco": ["baixo"] * n,
        }
    ).to_csv(csv_path, index=False)

    contract = build_contract(schema_path=schema_path, anchor_csv=csv_path)

    assert contract.correlations == []
    assert any("indefinida" in w for w in contract.warnings)


def test_build_contract_categoria_inesperada_gera_aviso(
    tmp_path: Path, schema_path: Path
) -> None:
    csv_path = tmp_path / "dados_categoria_extra.csv"
    n = 100
    pd.DataFrame(
        {
            "idade": np.linspace(30, 50, n),
            "rendimento": np.linspace(1000, 5000, n),
            "risco": ["baixo"] * (n - 1) + ["desconhecido"],
        }
    ).to_csv(csv_path, index=False)

    contract = build_contract(schema_path=schema_path, anchor_csv=csv_path)

    assert any("categorias não declaradas" in w for w in contract.warnings)


def test_build_contract_regra_de_negocio_violada_gera_aviso(
    tmp_path: Path, schema_path: Path
) -> None:
    csv_path = tmp_path / "dados_violam_regra.csv"
    n = 100
    rendimento = np.linspace(1000, 5000, n)
    rendimento[0] = -100  # viola "rendimento > 0"
    pd.DataFrame(
        {"idade": np.linspace(30, 50, n), "rendimento": rendimento, "risco": ["baixo"] * n}
    ).to_csv(csv_path, index=False)

    contract = build_contract(schema_path=schema_path, anchor_csv=csv_path)

    assert any("violada" in w for w in contract.warnings)


# --------------------------------------------------------------------------
# Pilar 1 / Teste 2 — parâmetros manuais contraditórios
# --------------------------------------------------------------------------


def test_build_contract_rejeita_parametros_manuais_contraditorios(
    schema_path: Path,
) -> None:
    # domínio de 'idade' é [30, 50]; média manual = 20 é contraditória
    manual_params = {
        "idade": {"distribuicao": "normal", "media": 20, "desvio": 5},
        "rendimento": {"distribuicao": "normal", "media": 3000, "desvio": 500},
        "risco": {
            "distribuicao": "categorica",
            "frequencias": {"baixo": 0.5, "medio": 0.3, "alto": 0.2},
        },
    }
    with pytest.raises(NeocortexValidationError, match="contraditório"):
        build_contract(schema_path=schema_path, manual_params=manual_params)


def test_build_contract_aceita_parametros_manuais_validos(schema_path: Path) -> None:
    manual_params = {
        "idade": {"distribuicao": "normal", "media": 40, "desvio": 5},
        "rendimento": {"distribuicao": "normal", "media": 3000, "desvio": 500},
        "risco": {
            "distribuicao": "categorica",
            "frequencias": {"baixo": 0.5, "medio": 0.3, "alto": 0.2},
        },
    }
    contract = build_contract(schema_path=schema_path, manual_params=manual_params)

    assert contract.source == ContractSource.MANUAL_PARAMS
    assert contract.is_synthetic_pure is False
    assert contract.column_stats["idade"].media == 40
    assert contract.correlations == []


def test_build_contract_rejeita_desvio_nao_positivo(schema_path: Path) -> None:
    manual_params = {
        "idade": {"distribuicao": "normal", "media": 40, "desvio": 0},
        "rendimento": {"distribuicao": "normal", "media": 3000, "desvio": 500},
        "risco": {
            "distribuicao": "categorica",
            "frequencias": {"baixo": 0.5, "medio": 0.3, "alto": 0.2},
        },
    }
    with pytest.raises(NeocortexValidationError, match="desvio"):
        build_contract(schema_path=schema_path, manual_params=manual_params)


def test_build_contract_rejeita_frequencias_categoria_fora_do_esquema(
    schema_path: Path,
) -> None:
    manual_params = {
        "idade": {"distribuicao": "normal", "media": 40, "desvio": 5},
        "rendimento": {"distribuicao": "normal", "media": 3000, "desvio": 500},
        "risco": {"distribuicao": "categorica", "frequencias": {"inexistente": 1.0}},
    }
    with pytest.raises(NeocortexValidationError, match="contraditório"):
        build_contract(schema_path=schema_path, manual_params=manual_params)


# --------------------------------------------------------------------------
# Pilar 1 / Teste 3 — Modo Sintético Puro
# --------------------------------------------------------------------------


def test_build_contract_sem_ancora_gera_alerta_e_prossegue(schema_path: Path) -> None:
    contract = build_contract(schema_path=schema_path)

    assert contract.is_synthetic_pure is True
    assert contract.source == ContractSource.SYNTHETIC_PURE
    assert any("MODO SINTÉTICO PURO" in w for w in contract.warnings)
    # prossegue com distribuições padrão em vez de falhar
    assert "idade" in contract.column_stats
    assert "risco" in contract.column_stats
    assert contract.column_stats["risco"].frequencias == pytest.approx(
        {"baixo": 1 / 3, "medio": 1 / 3, "alto": 1 / 3}
    )


def test_build_contract_sintetico_sem_dominio_usa_normal_padrao(tmp_path: Path) -> None:
    path = tmp_path / "schema_sem_dominio.yaml"
    path.write_text(
        yaml.safe_dump({"colunas": {"altura": {"tipo": "numerico"}}}), encoding="utf-8"
    )
    contract = build_contract(schema_path=path)

    assert contract.column_stats["altura"].media == 0.0
    assert contract.column_stats["altura"].desvio == 1.0
    assert any("normal-padrão" in w for w in contract.warnings)


def test_build_contract_sintetico_categorica_sem_categorias_levanta_erro(
    tmp_path: Path,
) -> None:
    path = tmp_path / "schema_invalido.yaml"
    path.write_text(
        yaml.safe_dump({"colunas": {"risco": {"tipo": "categorico"}}}), encoding="utf-8"
    )
    with pytest.raises(NeocortexValidationError, match="categorias"):
        build_contract(schema_path=path)


# --------------------------------------------------------------------------
# Imutabilidade e mútua exclusividade
# --------------------------------------------------------------------------


def test_build_contract_deteta_ordem_categorica_a_partir_do_csv(tmp_path: Path) -> None:
    """Regressão: sem `ordem_categorias`, o Módulo 2 mapeava percentis da
    numérica para categorias na ordem arbitrária de `value_counts`,
    desligada da associação real — um bug de fidelidade detectado
    durante a validação end-to-end do sistema. Este teste fixa a correção.
    """
    schema = {
        "colunas": {
            "score": {"tipo": "numerico"},
            "risco": {"tipo": "categorico", "categorias": ["baixo", "medio", "alto"]},
        },
        "regras": [],
    }
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")

    rng = np.random.default_rng(0)
    n = 300
    score = rng.normal(650, 90, n)
    # risco realmente depende de score: baixo score -> alto risco
    risco = np.where(score < 580, "alto", np.where(score < 700, "medio", "baixo"))
    # embaralha deliberadamente a frequência de "alto" para ser a categoria
    # mais comum (garante que a ordem de value_counts() NÃO coincide com
    # a ordem real por média crescente, expondo o bug se reaparecer)
    csv_path = tmp_path / "dados.csv"
    pd.DataFrame({"score": score, "risco": risco}).to_csv(csv_path, index=False)

    contract = build_contract(schema_path=path, anchor_csv=csv_path)
    stat = contract.column_stats["risco"]

    assert stat.driver_numerico == "score"
    assert stat.ordem_categorias == ["alto", "medio", "baixo"]  # ordem por média CRESCENTE de score


def test_build_contract_sem_associacao_numerica_ordem_categorias_e_none(tmp_path: Path) -> None:
    schema = {
        "colunas": {
            "score": {"tipo": "numerico"},
            "risco": {"tipo": "categorico", "categorias": ["baixo", "medio", "alto"]},
        },
        "regras": [],
    }
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")

    rng = np.random.default_rng(1)
    n = 200
    # risco INDEPENDENTE de score (sem associação real)
    df = pd.DataFrame({"score": rng.normal(650, 90, n), "risco": rng.choice(["baixo", "medio", "alto"], n)})
    csv_path = tmp_path / "dados.csv"
    df.to_csv(csv_path, index=False)

    contract = build_contract(schema_path=path, anchor_csv=csv_path)
    stat = contract.column_stats["risco"]

    assert stat.driver_numerico is None
    assert stat.ordem_categorias is None


def test_contract_e_imutavel(schema_path: Path) -> None:
    contract = build_contract(schema_path=schema_path)
    with pytest.raises(Exception):
        contract.is_synthetic_pure = False  # type: ignore[misc]


def test_build_contract_rejeita_csv_e_manual_em_simultaneo(
    tmp_path: Path, schema_path: Path
) -> None:
    csv_path = tmp_path / "dados.csv"
    _make_correlated_csv(csv_path, n_rows=60, n_cols=3)
    with pytest.raises(NeocortexValidationError, match="apenas uma âncora"):
        build_contract(
            schema_path=schema_path,
            anchor_csv=csv_path,
            manual_params={"idade": {"media": 40, "desvio": 5}},
        )


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end
# --------------------------------------------------------------------------


def test_pipeline_completo_end_to_end_csv_real(
    tmp_path: Path, large_schema_path: Path
) -> None:
    """Testa o módulo como um todo: esquema -> CSV real -> contrato final."""
    csv_path = tmp_path / "dados_e2e.csv"
    _make_correlated_csv(csv_path, n_rows=300, n_cols=10)

    contract = build_contract(
        schema_path=large_schema_path,
        rules=[],
        anchor_csv=csv_path,
        tolerancia_kl=0.05,
    )

    assert contract.source == ContractSource.REAL_CSV
    assert contract.tolerancia_kl == 0.05
    assert contract.sample_size == 300
    assert len(contract.column_stats) == 10
    assert all(
        stat.quartis[0] <= stat.quartis[1] <= stat.quartis[2]
        for stat in contract.column_stats.values()
    )
    # o contrato serializa corretamente em JSON (para consumo pelos módulos seguintes)
    serialized = contract.model_dump_json()
    assert "correlations" in serialized
