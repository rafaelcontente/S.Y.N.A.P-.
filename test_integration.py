"""Testes de integração end-to-end do S.Y.N.A.P. — todos os 7 módulos reais.

Ao contrário dos testes unitários de cada módulo (que usam contratos e
dados construídos à mão para isolar comportamento), estes testes correm
o pipeline completo (`main.run_synap`) através de `neocortex.build_contract`
até `compilador.run_compilador`, com todos os módulos reais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from synap.main import SynapConfig, run_synap

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def schema_path(tmp_path):
    """Esquema com uma coluna categórica dependente de duas numéricas correlacionadas."""
    schema = {
        "colunas": {
            "idade": {"tipo": "numerico", "minimo": 18, "maximo": 75},
            "rendimento": {"tipo": "numerico", "minimo": 0},
            "risco": {"tipo": "categorico", "categorias": ["baixo", "medio", "alto"]},
        },
        "regras": ["rendimento > 0"],
    }
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    return path


@pytest.fixture
def anchor_csv_path(tmp_path):
    """CSV de ancoragem: idade e rendimento correlacionados (~0.6), risco dependente de rendimento."""
    rng = np.random.default_rng(0)
    n = 300
    z_idade = rng.normal(0, 1, n)
    z_rendimento = 0.6 * z_idade + np.sqrt(1 - 0.6**2) * rng.normal(0, 1, n)
    idade = np.clip(45 + 15 * z_idade, 18, 75)
    rendimento = np.clip(3000 + 1200 * z_rendimento, 100, None)
    risco = np.where(rendimento > 4000, "alto", np.where(rendimento > 2000, "medio", "baixo"))
    df = pd.DataFrame({"idade": idade, "rendimento": rendimento, "risco": risco})
    path = tmp_path / "dados_reais.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def real_holdout_path(tmp_path, anchor_csv_path):
    """Reaproveita o mesmo processo gerador do anchor para o holdout do Módulo 7."""
    rng = np.random.default_rng(99)
    n = 200
    z_idade = rng.normal(0, 1, n)
    z_rendimento = 0.6 * z_idade + np.sqrt(1 - 0.6**2) * rng.normal(0, 1, n)
    idade = np.clip(45 + 15 * z_idade, 18, 75)
    rendimento = np.clip(3000 + 1200 * z_rendimento, 100, None)
    risco = np.where(rendimento > 4000, "alto", np.where(rendimento > 2000, "medio", "baixo"))
    df = pd.DataFrame({"idade": idade, "rendimento": rendimento, "risco": risco})
    path = tmp_path / "holdout_real.csv"
    df.to_csv(path, index=False)
    return path


def test_run_synap_pipeline_completo_com_auditor(
    tmp_path, schema_path, anchor_csv_path, real_holdout_path
) -> None:
    """Corre os 8 módulos reais: quando o Compilador liberta a saída, o
    Auditor (Módulo 8) corre automaticamente e produz o seu próprio veredito.
    """
    config = SynapConfig(
        schema_path=schema_path,
        anchor_csv=anchor_csv_path,
        seed_rows=5000,
        target_rows=300,
        batch_size=300,
        real_holdout_csv=real_holdout_path,
        target_column="risco",
        output_dir=tmp_path / "outputs",
        forcar_saida=True,  # garante libertação, para exercitar o Módulo 8
        rng_seed=0,
    )

    resultado = run_synap(config)

    assert resultado.resultado_compilador.liberado is True
    assert resultado.resultado_auditor is not None
    assert resultado.resultado_auditor.report_path is not None
    from pathlib import Path as _Path

    assert _Path(resultado.resultado_auditor.report_path).exists()
    assert resultado.resultado_auditor.utilidade.tarefas  # pelo menos uma tarefa avaliada


# --------------------------------------------------------------------------
# Teste de integração — pipeline completo end-to-end
# --------------------------------------------------------------------------


def test_run_synap_pipeline_completo_com_holdout_real(
    tmp_path, schema_path, anchor_csv_path, real_holdout_path
) -> None:
    """Corre TODOS os 7 módulos reais, do esquema ao CSV final + relatório."""
    config = SynapConfig(
        schema_path=schema_path,
        anchor_csv=anchor_csv_path,
        seed_rows=5000,
        target_rows=300,
        batch_size=150,
        real_holdout_csv=real_holdout_path,
        target_column="risco",
        output_dir=tmp_path / "outputs",
        rng_seed=0,
    )

    resultado = run_synap(config)

    assert resultado.pausado_por_homeostasia is False
    assert resultado.total_linhas_geradas == 300
    assert len(resultado.lotes) == 2  # 300 linhas / 150 por lote
    assert resultado.contract.column_stats.keys() == {"idade", "rendimento", "risco"}
    assert resultado.resultado_compilador is not None

    # invariante central: o dataset final respeita SEMPRE a regra de negócio,
    # mesmo depois de atravessar Núcleo + Hipotálamo + Homeostasia
    saida = resultado.resultado_compilador
    assert saida.validacao_asp.aprovado is True

    if saida.liberado:
        assert saida.csv_path is not None
        df_final = pd.read_csv(saida.csv_path)
        assert len(df_final) == 300
        assert (df_final["rendimento"] > 0).all()
        assert saida.report_path is not None
    else:
        # mesmo bloqueado (fidelidade insuficiente com um dataset tão
        # pequeno é plausível), o diagnóstico tem de ser explicativo
        assert saida.diagnostico is not None


def test_run_synap_sem_holdout_bloqueia_por_omissao(tmp_path, schema_path, anchor_csv_path) -> None:
    config = SynapConfig(
        schema_path=schema_path,
        anchor_csv=anchor_csv_path,
        seed_rows=5000,
        target_rows=200,
        batch_size=200,
        output_dir=tmp_path / "outputs",
        rng_seed=1,
    )

    resultado = run_synap(config)

    assert resultado.pausado_por_homeostasia is False
    assert resultado.resultado_compilador.liberado is False
    assert resultado.resultado_compilador.fidelidade.indice_fidelidade is None


def test_run_synap_forcar_saida_liberta_sem_holdout(tmp_path, schema_path, anchor_csv_path) -> None:
    config = SynapConfig(
        schema_path=schema_path,
        anchor_csv=anchor_csv_path,
        seed_rows=5000,
        target_rows=200,
        batch_size=200,
        output_dir=tmp_path / "outputs",
        forcar_saida=True,
        rng_seed=1,
    )

    resultado = run_synap(config)

    assert resultado.resultado_compilador.liberado is True
    assert (tmp_path / "outputs" / "dataset_final.csv").exists()


def test_run_synap_manual_params_sem_anchor_csv(tmp_path, schema_path) -> None:
    """Executa sem CSV de ancoragem (parâmetros manuais) — caminho alternativo do Módulo 1."""
    manual_params = {
        "idade": {"distribuicao": "normal", "media": 45, "desvio": 15},
        "rendimento": {"distribuicao": "normal", "media": 3000, "desvio": 1200},
        "risco": {
            "distribuicao": "categorica",
            "frequencias": {"baixo": 0.4, "medio": 0.4, "alto": 0.2},
        },
    }
    config = SynapConfig(
        schema_path=schema_path,
        manual_params=manual_params,
        seed_rows=5000,
        target_rows=150,
        batch_size=150,
        output_dir=tmp_path / "outputs",
        forcar_saida=True,
        rng_seed=2,
    )

    resultado = run_synap(config)

    assert resultado.total_linhas_geradas == 150
    assert resultado.resultado_compilador.liberado is True


def test_synap_config_rejeita_schema_inexistente(tmp_path) -> None:
    config = SynapConfig(
        schema_path=tmp_path / "nao_existe.yaml",
        manual_params={"idade": {"media": 40, "desvio": 10}},
        target_rows=100,
    )
    with pytest.raises(Exception):
        run_synap(config)


# --------------------------------------------------------------------------
# Website local (synap --serve-site) — liga o sistema ao website sem
# depender de nenhum repositório remoto já existir
# --------------------------------------------------------------------------


def test_serve_site_rejeita_repo_sem_website(tmp_path) -> None:
    from synap.main import serve_site

    with pytest.raises(FileNotFoundError):
        serve_site(repo_root=tmp_path, port=0, open_browser=False)


def test_serve_site_serve_a_raiz_do_repositorio_e_resolve_ligacoes_relativas():
    """Testa o mecanismo real de synap --serve-site: a raiz do
    repositório é servida (não só site/), para que as ligações
    relativas do website (../README.md, ../CONTRIBUTING.md, etc.)
    resolvam sem depender de o projeto estar publicado no GitHub.
    """
    import threading
    import time
    import urllib.request

    from synap.main import REPO_ROOT, serve_site

    port = 8765
    thread = threading.Thread(
        target=serve_site, kwargs={"port": port, "open_browser": False}, daemon=True
    )
    thread.start()
    time.sleep(1.0)

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/site/", timeout=5) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "S.Y.N.A.P." in html

        # a mesma ligação relativa que o website usa (../README.md a
        # partir de site/index.html) tem de resolver para o README real
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/README.md", timeout=5) as resp:
            assert resp.status == 200
            conteudo = resp.read()
            assert conteudo == (REPO_ROOT / "README.md").read_bytes()
    finally:
        pass  # servidor em thread daemon; termina com o processo de teste
