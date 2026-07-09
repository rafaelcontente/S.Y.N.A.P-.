"""Servidor web do S.Y.N.A.P. — liga o sistema real ao website.

Ao contrário de uma reimplementação em JavaScript, este módulo expõe o
pipeline REAL (`synap.main.run_synap`, os 8 módulos Python completos)
através de uma pequena API HTTP, para que a geração de dados possa ser
acionada diretamente a partir de `site/index.html` — o browser envia o
esquema e os ficheiros, o Python corre os 8 módulos, o browser recebe
o resultado real (Índice de Fidelidade, veredito de privacidade, e os
ficheiros gerados de facto).

Os pedidos de geração correm em threads de fundo, identificados por um
`job_id`; o website faz *polling* a `/api/jobs/<job_id>` até o job
terminar. Isto evita bloquear o pedido HTTP inicial durante os
(potencialmente vários segundos a minutos) que o pipeline demora.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from synap.main import REPO_ROOT, SynapConfig, run_synap

logger = logging.getLogger(__name__)

MAX_TARGET_ROWS_WEB: int = 5_000
"""Teto de linhas por pedido vindo do browser — protege o servidor
local de pedidos acidentalmente enormes através da UI."""

MIN_SEED_ROWS_WEB: int = 5_000
"""O Módulo 2 exige uma semente entre 5 000 e 10 000 linhas; a UI web
usa sempre o mínimo, para manter os tempos de resposta razoáveis."""

JOB_WORKDIR: Path = Path(tempfile.gettempdir()) / "synap_web_jobs"
JOB_WORKDIR.mkdir(exist_ok=True)

JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


@dataclass
class Job:
    """Estado de uma execução do pipeline pedida a partir do website."""

    id: str
    status: str = "a_iniciar"  # a_iniciar | a_correr | concluido | bloqueado | pausado | erro
    log: list[str] = field(default_factory=list)
    resultado: dict[str, Any] | None = None
    erro: str | None = None
    output_dir: Path | None = None


def _resumo_lote(lote: Any) -> dict[str, Any]:
    return {
        "indice_lote": lote.indice_lote,
        "linhas_geradas": lote.linhas_geradas,
        "linhas_acumuladas": lote.linhas_acumuladas,
        "derivas_causais_detectadas": lote.derivas_causais_detectadas,
        "regras_ilp_consolidadas": lote.regras_ilp_consolidadas,
        "estado_homeostase": lote.estado_homeostase,
        "artefactos_detectados": lote.artefactos_detectados,
    }


def _run_job(job: Job, config: SynapConfig) -> None:
    """Corre o pipeline real numa thread de fundo, atualizando o estado do job."""
    job.status = "a_correr"
    job.log.append("Módulo 1 (Percetor): a construir o Contrato Estatístico…")
    try:
        resultado = run_synap(config)
        job.output_dir = config.output_dir

        for lote in resultado.lotes:
            job.log.append(
                f"Lote {lote.indice_lote}: {lote.linhas_geradas} linhas aprovadas "
                f"(total {lote.linhas_acumuladas}) — homeostasia: {lote.estado_homeostase}"
            )
        job.log.extend(resultado.warnings)

        if resultado.pausado_por_homeostasia:
            job.status = "pausado"
            job.resultado = {
                "pausado_por_homeostasia": True,
                "total_linhas_geradas": resultado.total_linhas_geradas,
                "lotes": [_resumo_lote(lote) for lote in resultado.lotes],
                "warnings": resultado.warnings,
            }
            return

        comp = resultado.resultado_compilador
        aud = resultado.resultado_auditor
        job.resultado = {
            "pausado_por_homeostasia": False,
            "total_linhas_geradas": resultado.total_linhas_geradas,
            "lotes": [_resumo_lote(lote) for lote in resultado.lotes],
            "compilador": (
                {
                    "liberado": comp.liberado,
                    "diagnostico": comp.diagnostico,
                    "asp_aprovadas": comp.validacao_asp.aprovadas,
                    "asp_total": comp.validacao_asp.total_linhas,
                    "indice_fidelidade": (comp.fidelidade.indice_fidelidade if comp.fidelidade else None),
                    "coluna_alvo": (comp.fidelidade.coluna_alvo if comp.fidelidade else None),
                    "csv_path": comp.csv_path,
                    "report_path": comp.report_path,
                }
                if comp
                else None
            ),
            "auditor": (
                {
                    "aprovado_global": aud.aprovado_global,
                    "retencao_utilidade": aud.utilidade.retencao_media,
                    "privacidade_avaliavel": aud.privacidade.avaliavel,
                    "privacidade_aprovada": aud.privacidade.aprovado,
                    "privacidade_motivo": aud.privacidade.motivo,
                    "report_path": aud.report_path,
                }
                if aud
                else None
            ),
            "warnings": resultado.warnings,
        }
        job.status = "concluido" if (comp and comp.liberado) else "bloqueado"
    except Exception as exc:  # noqa: BLE001 — reportado ao website, não é suposto derrubar o servidor
        logger.exception("Erro no job %s", job.id)
        job.status = "erro"
        job.erro = f"{type(exc).__name__}: {exc}"


def create_app(repo_root: Path = REPO_ROOT) -> Flask:
    """Constrói a aplicação Flask: website + API do pipeline real."""
    app = Flask(__name__, static_folder=None)
    site_dir = repo_root / "site"

    # --------------------------------------------------------------
    # Website e documentos do repositório (ligações relativas reais)
    # --------------------------------------------------------------
    @app.get("/")
    @app.get("/site/")
    def _root():
        return send_from_directory(site_dir, "index.html")

    @app.get("/site/<path:filename>")
    def _site_files(filename: str):
        return send_from_directory(site_dir, filename)

    @app.get("/<path:filename>")
    def _repo_files(filename: str):
        # Serve documentos reais da raiz do repositório (README,
        # PROJECT_NARRATIVE, validation/*.pdf, etc.) — as mesmas
        # ligações relativas ("../README.md") que o website usa.
        full = (repo_root / filename).resolve()
        if repo_root.resolve() not in full.parents:
            abort(403)
        if not full.is_file():
            abort(404)
        return send_from_directory(full.parent, full.name)

    # --------------------------------------------------------------
    # API do pipeline real
    # --------------------------------------------------------------
    @app.post("/api/generate")
    def _api_generate():
        try:
            payload = json.loads(request.form.get("config", "{}"))
        except json.JSONDecodeError:
            return jsonify({"erro": "campo 'config' não é JSON válido"}), 400

        if "schema" not in payload or "colunas" not in payload["schema"]:
            return jsonify({"erro": "payload sem 'schema.colunas'"}), 400

        job_id = uuid.uuid4().hex[:12]
        job_dir = JOB_WORKDIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        schema_path = job_dir / "schema.json"
        schema_path.write_text(json.dumps(payload["schema"]), encoding="utf-8")

        def _save_upload(field_name: str) -> Path | None:
            file = request.files.get(field_name)
            if not file or not file.filename:
                return None
            dest = job_dir / secure_filename(file.filename)
            file.save(dest)
            return dest

        anchor_csv = _save_upload("anchor_csv")
        real_holdout_csv = _save_upload("real_holdout_csv")
        training_reference_csv = _save_upload("training_reference_csv")

        target_rows = min(int(payload.get("target_rows", 500)), MAX_TARGET_ROWS_WEB)

        try:
            config = SynapConfig(
                schema_path=schema_path,
                anchor_csv=anchor_csv,
                manual_params=payload.get("manual_params") if anchor_csv is None else None,
                rules=payload["schema"].get("regras"),
                seed_rows=MIN_SEED_ROWS_WEB,
                target_rows=target_rows,
                batch_size=min(target_rows, 2_500),
                real_holdout_csv=real_holdout_csv,
                training_reference_csv=training_reference_csv,
                target_column=payload.get("target_column") or None,
                fidelity_threshold=float(payload.get("fidelity_threshold", 0.90)),
                run_auditor_step=bool(payload.get("run_auditor", True)),
                output_dir=job_dir / "saida",
                forcar_saida=bool(payload.get("forcar_saida", False)),
                rng_seed=payload.get("rng_seed"),
            )
        except Exception as exc:  # noqa: BLE001 — erro de validação da configuração, devolvido ao browser
            return jsonify({"erro": f"configuração inválida: {exc}"}), 400

        job = Job(id=job_id)
        with _JOBS_LOCK:
            JOBS[job_id] = job
        threading.Thread(target=_run_job, args=(job, config), daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.get("/api/jobs/<job_id>")
    def _api_job_status(job_id: str):
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        return jsonify(
            {"status": job.status, "log": job.log, "resultado": job.resultado, "erro": job.erro}
        )

    @app.get("/api/jobs/<job_id>/download/<name>")
    def _api_download(job_id: str, name: str):
        job = JOBS.get(job_id)
        if job is None or job.output_dir is None:
            abort(404)
        return send_from_directory(job.output_dir, secure_filename(name), as_attachment=True)

    return app
