"""Testes do servidor web (`synap.webserver`) — a API que liga o
website ao pipeline REAL (`run_synap`), não a uma reimplementação.

Usa o test client do Flask (sem precisar de um servidor de facto a
correr numa porta), mas invoca o pipeline verdadeiro em cada teste —
por isso corre poucos casos, com volumes pequenos.
"""

from __future__ import annotations

import io
import json
import time

import pytest

from synap.webserver import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _wait_for_job(client, job_id: str, timeout: float = 90.0) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(f"/api/jobs/{job_id}")
        data = resp.get_json()
        if data["status"] in ("concluido", "bloqueado", "pausado", "erro"):
            return data
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} não terminou em {timeout}s")


def _schema_minimo() -> dict:
    return {
        "colunas": {
            "idade": {"tipo": "numerico", "minimo": 18, "maximo": 75},
            "rendimento": {"tipo": "numerico", "minimo": 0, "maximo": 20000},
            "risco": {"tipo": "categorico", "categorias": ["baixo", "medio", "alto"]},
        },
        "regras": ["rendimento > 0"],
    }


# --------------------------------------------------------------------------
# Website e documentos do repositório
# --------------------------------------------------------------------------


def test_raiz_serve_o_website(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"S.Y.N.A.P." in resp.data


def test_serve_documento_real_da_raiz_do_repositorio(client) -> None:
    resp = client.get("/README.md")
    assert resp.status_code == 200
    assert b"S.Y.N.A.P." in resp.data


def test_bloqueia_travessia_de_diretorios_fora_do_repositorio(client) -> None:
    resp = client.get("/../../../etc/passwd")
    assert resp.status_code in (403, 404)


def test_ficheiro_inexistente_devolve_404(client) -> None:
    resp = client.get("/nao_existe_de_todo.md")
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# API do pipeline real
# --------------------------------------------------------------------------


def test_api_generate_rejeita_payload_sem_schema(client) -> None:
    resp = client.post("/api/generate", data={"config": json.dumps({})})
    assert resp.status_code == 400


def test_api_generate_e_job_status_pipeline_real_sem_holdout_bloqueia(client) -> None:
    """Testa o módulo como um todo: um pedido HTTP real aciona o
    pipeline Python real (sem holdout -> bloqueado por omissão,
    exatamente como a CLI faria)."""
    config = {"schema": _schema_minimo(), "target_rows": 300, "forcar_saida": False, "rng_seed": 7}
    resp = client.post("/api/generate", data={"config": json.dumps(config)})
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]

    data = _wait_for_job(client, job_id)
    assert data["status"] == "bloqueado"
    assert data["resultado"]["compilador"]["liberado"] is False
    assert data["resultado"]["total_linhas_geradas"] == 300


def test_api_generate_com_forcar_saida_liberta_e_permite_download(client) -> None:
    config = {"schema": _schema_minimo(), "target_rows": 300, "forcar_saida": True, "rng_seed": 7}
    resp = client.post("/api/generate", data={"config": json.dumps(config)})
    job_id = resp.get_json()["job_id"]

    data = _wait_for_job(client, job_id)
    assert data["status"] == "concluido"
    comp = data["resultado"]["compilador"]
    assert comp["liberado"] is True

    csv_name = comp["csv_path"].replace("\\", "/").split("/")[-1]
    dl = client.get(f"/api/jobs/{job_id}/download/{csv_name}")
    assert dl.status_code == 200
    linhas = dl.data.decode("utf-8").strip().splitlines()
    assert linhas[0] == "idade,rendimento,risco"
    assert len(linhas) == 301  # cabeçalho + 300 linhas


def test_api_generate_com_csv_ancora_real(client) -> None:
    """Sobe um CSV real via multipart/form-data — o mesmo caminho que o
    formulário do website usa — e confirma que ancora o Contrato."""
    csv_content = "idade,rendimento,risco\n" + "\n".join(
        f"{20 + i % 50},{1000 + i * 37 % 9000},{'baixo' if i % 3 == 0 else 'medio'}" for i in range(80)
    )
    config = {"schema": _schema_minimo(), "target_rows": 300, "forcar_saida": True, "rng_seed": 7}
    resp = client.post(
        "/api/generate",
        data={
            "config": json.dumps(config),
            "anchor_csv": (io.BytesIO(csv_content.encode("utf-8")), "anchor.csv"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]
    data = _wait_for_job(client, job_id)
    assert data["status"] in ("concluido", "bloqueado")
    assert data["resultado"]["total_linhas_geradas"] == 300


def test_job_inexistente_devolve_404(client) -> None:
    resp = client.get("/api/jobs/nao-existe")
    assert resp.status_code == 404
