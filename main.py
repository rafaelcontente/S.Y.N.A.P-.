"""S.Y.N.A.P. — Orquestrador principal.

Integra os 8 módulos do sistema num único fluxo:

    Módulo 1 (neocortex)   -> Contrato Estatístico
    Módulo 2 (hipocampo)   -> DAG ajustada + Semente
    "Loop cognitivo"       -> por lote (= 1 checkpoint):
        Núcleo (nucleo)        -> remistura + Mahalanobis/plausibilidade (M3)
                                   + ASP linha-a-linha (M4) + autoencoder (M5)
        Hipotálamo (hipotalamo) -> deriva causal contínua + indução de regras (M4)
        Homeostase (homeostase) -> KL-divergência + entropia, reponderação (M6)
    Módulo 7 (compilador)  -> validação ASP final + Modelo Gémeo + CSV/relatório
    Módulo 8 (auditor)     -> utilidade TSTR/TRTR + privacidade (DCR/NNDR/MIA)

Ver o histórico de construção módulo-a-módulo para as decisões de
arquitectura por trás de cada peça; este ficheiro apenas as compõe.
"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from synap import auditor, compilador, hipocampo, hipotalamo, homeostase, neocortex, nucleo

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

DEFAULT_SEED_ROWS: int = 5_000
DEFAULT_BATCH_SIZE: int = 5_000
"""Também usado como `checkpoint_size` do Módulo 4 e cadência do Módulo 6 —
um lote do Núcleo corresponde a exatamente um checkpoint de ambos."""

DEFAULT_FIDELITY_THRESHOLD: float = 0.90
MAX_LOOP_ITERATIONS_MULTIPLIER: int = 20
"""Orçamento defensivo de iterações do loop principal (evita ciclo sem
fim se a geração estagnar): `target_rows / batch_size * este_fator`."""


# --------------------------------------------------------------------------
# Configuração e resultado
# --------------------------------------------------------------------------


class SynapConfig(BaseModel):
    """Configuração de uma execução completa do S.Y.N.A.P.

    Attributes:
        schema_path: Caminho do esquema YAML/JSON (Módulo 1).
        rules: Regras de negócio em texto (sobrepõe as do esquema, se dadas).
        anchor_csv: CSV de dados reais para ancoragem estatística (Módulo 1).
        manual_params: Parâmetros estatísticos manuais (alternativa a `anchor_csv`).
        user_dag_edges: DAG causal fornecida pelo utilizador (Módulo 2).
        causal_order: Ordem causal para desambiguar arestas sem DAG explícita.
        seed_rows: Nº de linhas da semente inicial (Módulo 2).
        target_rows: Nº total de linhas a gerar no dataset final.
        batch_size: Tamanho de cada lote do Núcleo == `checkpoint_size` dos Módulos 4/6.
        fidelity_threshold: Índice de Fidelidade mínimo para libertar a saída (Módulo 7).
        real_holdout_csv: CSV de dados reais para o Índice de Fidelidade (Módulo 7).
        target_column: Coluna-alvo do Modelo Gémeo (Módulo 7); se `None`, a
            de maior variância é escolhida automaticamente.
        output_dir: Diretório de saída do CSV final e do relatório PDF.
        forcar_saida: Se True, liberta a saída mesmo com fidelidade baixa
            ou sem holdout real (ver Módulo 7).
        rng_seed: Semente aleatória, para reprodutibilidade de toda a execução.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_path: Path
    rules: list[str] | None = None
    anchor_csv: Path | None = None
    manual_params: dict[str, dict[str, Any]] | None = None
    user_dag_edges: list[tuple[str, str]] | None = None
    causal_order: list[str] | None = None

    seed_rows: int = DEFAULT_SEED_ROWS
    target_rows: int
    batch_size: int = DEFAULT_BATCH_SIZE

    fidelity_threshold: float = DEFAULT_FIDELITY_THRESHOLD
    real_holdout_csv: Path | None = None
    target_column: str | None = None

    run_auditor_step: bool = True
    """Se True (omissão), corre o Módulo 8 (Auditor) após o Módulo 7
    libertar a saída — só faz sentido auditar um dataset já libertado."""
    training_reference_csv: Path | None = None
    """Amostra dos dados reais usados para ancorar o gerador (Módulo 1),
    distinta de `real_holdout_csv` — necessária apenas para o Ataque de
    Inferência de Pertença do Módulo 8."""
    utility_threshold: float = auditor.DEFAULT_UTILITY_RETENTION_THRESHOLD

    output_dir: Path = Path("/mnt/user-data/outputs")
    forcar_saida: bool = False
    rng_seed: int | None = None


class BatchSummary(BaseModel):
    """Resumo de um único lote do loop cognitivo, para o relatório de execução."""

    model_config = ConfigDict(frozen=True)

    indice_lote: int
    linhas_geradas: int
    linhas_acumuladas: int
    derivas_causais_detectadas: int
    regras_ilp_consolidadas: int
    estado_homeostase: str
    artefactos_detectados: int


class SynapResult(BaseModel):
    """Resultado completo de uma execução do S.Y.N.A.P.

    Attributes:
        contract: O Contrato Estatístico produzido pelo Módulo 1.
        dag_final: Representação textual (arestas) da DAG final do Módulo 2.
        lotes: Resumo de cada lote processado no loop cognitivo.
        total_linhas_geradas: Total de linhas efetivamente geradas (antes da compilação).
        pausado_por_homeostasia: True se o Módulo 6 recomendou pausa e o
            loop foi interrompido antes de atingir `target_rows`.
        resultado_compilador: A saída do Módulo 7 (`None` se pausado por homeostasia).
        warnings: Alertas agregados de toda a execução.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    contract: neocortex.StatisticalContract
    dag_final: list[tuple[str, str]]
    lotes: list[BatchSummary]
    total_linhas_geradas: int
    pausado_por_homeostasia: bool
    resultado_compilador: compilador.CompilerOutput | None
    resultado_auditor: auditor.AuditorOutput | None = None
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def run_synap(config: SynapConfig) -> SynapResult:
    """Executa o pipeline completo do S.Y.N.A.P., dos 7 módulos ao CSV final.

    Args:
        config: A configuração da execução (ver :class:`SynapConfig`).

    Returns:
        :class:`SynapResult` com o Contrato, o resumo de cada lote, e o
        resultado do Compilador (ou a indicação de pausa por homeostasia).
    """
    warnings: list[str] = []

    # --- Módulo 1: Contrato Estatístico ---
    contract = neocortex.build_contract(
        schema_path=config.schema_path,
        rules=config.rules,
        anchor_csv=config.anchor_csv,
        manual_params=config.manual_params,
    )
    warnings.extend(f"[M1] {w}" for w in contract.warnings)

    # --- Módulo 2: DAG + Semente ---
    saida_hipocampo = hipocampo.run_hipocampo(
        contract,
        user_dag_edges=config.user_dag_edges,
        causal_order=config.causal_order,
        n_rows=config.seed_rows,
        rng_seed=config.rng_seed,
    )
    warnings.extend(f"[M2] {w}" for w in saida_hipocampo.warnings)
    dag = saida_hipocampo.dag

    # --- Estado do loop cognitivo ---
    estado_hipotalamo = hipotalamo.HipotalamoState(contract)
    estado_homeostase = homeostase.HomeostaseState(contract)

    memoria_corrente = saida_hipocampo.seed
    pesos_correntes = np.ones(len(memoria_corrente))

    lotes_gerados: list[pd.DataFrame] = []
    resumos: list[BatchSummary] = []
    anti_exemplos_acumulados: list[Any] = []
    total_gerado = 0
    pausado = False

    max_iteracoes = max(1, (config.target_rows // config.batch_size) * MAX_LOOP_ITERATIONS_MULTIPLIER)
    indice_lote = 0

    while total_gerado < config.target_rows and indice_lote < max_iteracoes:
        indice_lote += 1
        n_rows_lote = min(config.batch_size, config.target_rows - total_gerado)

        # --- Núcleo: M3 (remistura+filtros) + M4 (ASP linha-a-linha) + M5 (autoencoder) ---
        saida_nucleo = nucleo.run_nucleo(
            seed=memoria_corrente,
            contract=contract,
            n_rows=n_rows_lote,
            dag=dag,
            initial_weights=pesos_correntes,
            rng_seed=config.rng_seed,
        )
        warnings.extend(f"[Núcleo/lote {indice_lote}] {w}" for w in saida_nucleo.warnings)

        if len(saida_nucleo.lote) == 0:
            warnings.append(
                f"[Núcleo/lote {indice_lote}] nenhuma linha aprovada neste lote — "
                "interrompendo o loop para evitar ciclo sem progresso"
            )
            break

        lotes_gerados.append(saida_nucleo.lote)
        anti_exemplos_acumulados.extend(saida_nucleo.anti_exemplos)
        total_gerado += len(saida_nucleo.lote)

        # --- Módulo 4: deriva causal contínua + indução de regras (ILP) ---
        saida_hipotalamo = hipotalamo.run_hipotalamo(
            saida_nucleo.lote,
            contract,
            dag,
            state=estado_hipotalamo,
            checkpoint_size=config.batch_size,
            rng_seed=config.rng_seed,
        )
        n_derivas = (
            len(saida_hipotalamo.monitorizacao_causal.derivas_detectadas)
            if saida_hipotalamo.monitorizacao_causal
            else 0
        )
        if n_derivas > 0:
            warnings.append(
                f"[M4/lote {indice_lote}] DERIVA CAUSAL: {n_derivas} par(es) afetado(s) "
                "— a DAG não é reajustada automaticamente a meio da geração; "
                "considere parar e reconstruir o Módulo 2 se persistir"
            )
        if saida_hipotalamo.novas_regras:
            warnings.append(
                f"[M4/lote {indice_lote}] {len(saida_hipotalamo.novas_regras)} regra(s) "
                "induzida(s) pelo ILP (não aplicadas automaticamente ao ASP — "
                "ver notas de integração; disponíveis para consolidação manual "
                "no Contrato)"
            )

        # --- Módulo 6: homeostasia (KL-divergência + entropia) ---
        saida_homeostase = homeostase.run_controlador_homeostasia(
            saida_nucleo.lote,
            contract,
            memory=saida_nucleo.memoria_final,
            state=estado_homeostase,
        )
        warnings.extend(f"[M6/lote {indice_lote}] {w}" for w in saida_homeostase.warnings)

        resumos.append(
            BatchSummary(
                indice_lote=indice_lote,
                linhas_geradas=len(saida_nucleo.lote),
                linhas_acumuladas=total_gerado,
                derivas_causais_detectadas=n_derivas,
                regras_ilp_consolidadas=len(saida_hipotalamo.novas_regras),
                estado_homeostase=saida_homeostase.relatorio.estado_global.value,
                artefactos_detectados=saida_nucleo.relatorio.artefactos_detectados,
            )
        )

        if saida_homeostase.pausa_recomendada:
            pausado = True
            warnings.append(
                f"[M6/lote {indice_lote}] PAUSA RECOMENDADA — divergência distribucional "
                "crítica. Geração interrompida antes de atingir target_rows."
            )
            break

        # --- pesos e memória para o próximo lote (composição multiplicativa) ---
        memoria_corrente = saida_nucleo.memoria_final
        pesos_correntes = np.array(saida_nucleo.pesos_finais) * np.array(saida_homeostase.pesos_multiplicador)

    if indice_lote >= max_iteracoes and total_gerado < config.target_rows:
        warnings.append(
            f"orçamento de {max_iteracoes} lotes esgotado antes de atingir "
            f"target_rows ({total_gerado}/{config.target_rows} linhas geradas)"
        )

    dataset_final = (
        pd.concat(lotes_gerados, ignore_index=True)
        if lotes_gerados
        else pd.DataFrame(columns=list(contract.column_stats))
    )
    dag_arestas = [(e.origem, e.destino) for e in dag.arestas]

    if pausado:
        return SynapResult(
            contract=contract,
            dag_final=dag_arestas,
            lotes=resumos,
            total_linhas_geradas=total_gerado,
            pausado_por_homeostasia=True,
            resultado_compilador=None,
            warnings=warnings,
        )

    # --- Módulo 7: validação ASP final + Modelo Gémeo + compilação ---
    real_holdout_df = pd.read_csv(config.real_holdout_csv) if config.real_holdout_csv else None
    resultado_compilador = compilador.run_compilador(
        dataset_final,
        contract,
        target_column=config.target_column,
        real_holdout=real_holdout_df,
        fidelity_threshold=config.fidelity_threshold,
        output_dir=config.output_dir,
        anti_examples=anti_exemplos_acumulados or None,
        forcar_saida=config.forcar_saida,
        rng_seed=config.rng_seed,
    )
    warnings.extend(f"[M7] {w}" for w in resultado_compilador.warnings)

    resultado_auditor: auditor.AuditorOutput | None = None
    if config.run_auditor_step and resultado_compilador.liberado and real_holdout_df is not None:
        training_reference_df = (
            pd.read_csv(config.training_reference_csv) if config.training_reference_csv else None
        )
        try:
            resultado_auditor = auditor.run_auditor(
                dataset_final,
                contract,
                real_holdout=real_holdout_df,
                training_reference=training_reference_df,
                utility_threshold=config.utility_threshold,
                output_dir=config.output_dir,
                rng_seed=config.rng_seed,
            )
            warnings.extend(f"[M8] {w}" for w in resultado_auditor.warnings)
            if not resultado_auditor.aprovado_global:
                warnings.append(f"[M8] {resultado_auditor.diagnostico}")
        except auditor.InsufficientHoldoutError as exc:
            warnings.append(f"[M8] auditoria não executada: {exc}")
    elif config.run_auditor_step and real_holdout_df is None:
        warnings.append(
            "[M8] auditoria não executada — sem 'real_holdout_csv' "
            "(utilidade e privacidade não podem ser provadas sem dados reais)"
        )

    logger.info(
        "S.Y.N.A.P. concluído: lotes=%d, linhas=%d, liberado=%s, auditoria_aprovada=%s",
        len(resumos),
        total_gerado,
        resultado_compilador.liberado,
        resultado_auditor.aprovado_global if resultado_auditor else None,
    )

    return SynapResult(
        contract=contract,
        dag_final=dag_arestas,
        lotes=resumos,
        total_linhas_geradas=total_gerado,
        pausado_por_homeostasia=False,
        resultado_compilador=resultado_compilador,
        resultado_auditor=resultado_auditor,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# CLI mínima
# --------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Raiz do repositório (dois níveis acima de `src/synap/main.py`) — usada
para localizar `site/` a partir de uma checkout local do código-fonte."""

DEFAULT_SITE_PORT: int = 8000


def serve_site(repo_root: Path = REPO_ROOT, port: int = DEFAULT_SITE_PORT, open_browser: bool = True) -> None:
    """Serve o website do projeto e a API do pipeline real, localmente.

    Liga o sistema ao website sem depender de nenhum repositório remoto
    (GitHub ou outro) já existir: serve os ficheiros locais desta
    checkout (incluindo `README.md`, `PROJECT_NARRATIVE.md`, o
    Relatório de Validação em `validation/`, etc., que o website
    referencia por caminho relativo), e expõe `/api/generate` +
    `/api/jobs/<id>` para que a geração de dados no website acione o
    pipeline REAL (`run_synap`, os 8 módulos Python) — não uma
    reimplementação em JavaScript.

    Args:
        repo_root: Raiz do repositório a servir (por omissão, a raiz
            detectada a partir desta instalação em modo editável).
        port: Porta local onde o servidor fica disponível.
        open_browser: Se True, abre automaticamente o browser no website (`/`).

    Raises:
        FileNotFoundError: Se `repo_root / "site" / "index.html"` não existir
            (ex.: a instalar via wheel/PyPI, onde `site/` não é empacotado —
            clone o repositório para usar esta funcionalidade).
    """
    index = repo_root / "site" / "index.html"
    if not index.exists():
        raise FileNotFoundError(
            f"'{index}' não existe. `serve_site` foi pensado para correr a partir "
            "de uma checkout local do repositório (ex.: `git clone` + `pip install -e .`), "
            "não a partir de um pacote instalado via wheel/PyPI."
        )

    from synap.webserver import create_app

    app = create_app(repo_root)
    url = f"http://127.0.0.1:{port}/"
    print(f"🌐 S.Y.N.A.P. — website + pipeline real em {url}  (Ctrl+C para parar)")
    print("   A geração de dados no website corre os 8 módulos Python de facto.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S.Y.N.A.P. — gerador de dados sintéticos")
    parser.add_argument(
        "--serve-site", action="store_true",
        help="Serve o website local do projeto (site/index.html) e sai — ignora os restantes argumentos de geração.",
    )
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT, help="Raiz do repositório a servir com --serve-site")
    parser.add_argument("--site-port", type=int, default=DEFAULT_SITE_PORT, help="Porta do servidor local do website")
    parser.add_argument("--no-open-browser", action="store_true", help="Não abrir o browser automaticamente com --serve-site")

    parser.add_argument("--schema", type=Path, default=None, help="Caminho do esquema YAML/JSON")
    parser.add_argument("--anchor-csv", type=Path, default=None, help="CSV de dados reais para ancoragem")
    parser.add_argument("--target-rows", type=int, default=None, help="Nº total de linhas a gerar")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed-rows", type=int, default=DEFAULT_SEED_ROWS)
    parser.add_argument("--real-holdout-csv", type=Path, default=None)
    parser.add_argument("--training-reference-csv", type=Path, default=None)
    parser.add_argument("--target-column", type=str, default=None)
    parser.add_argument("--fidelity-threshold", type=float, default=DEFAULT_FIDELITY_THRESHOLD)
    parser.add_argument("--utility-threshold", type=float, default=auditor.DEFAULT_UTILITY_RETENTION_THRESHOLD)
    parser.add_argument("--no-auditor", action="store_true", help="Desativa o Módulo 8 (Auditor)")
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/user-data/outputs"))
    parser.add_argument("--forcar-saida", action="store_true")
    parser.add_argument("--rng-seed", type=int, default=None)
    return parser


def main() -> None:
    """Ponto de entrada da CLI: `synap --schema ... --target-rows ...`, ou `synap --serve-site`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.serve_site:
        serve_site(repo_root=args.repo_dir, port=args.site_port, open_browser=not args.no_open_browser)
        return

    if args.schema is None or args.target_rows is None:
        parser.error("--schema e --target-rows são obrigatórios (a não ser que use --serve-site)")

    config = SynapConfig(
        schema_path=args.schema,
        anchor_csv=args.anchor_csv,
        target_rows=args.target_rows,
        batch_size=args.batch_size,
        seed_rows=args.seed_rows,
        real_holdout_csv=args.real_holdout_csv,
        training_reference_csv=args.training_reference_csv,
        target_column=args.target_column,
        fidelity_threshold=args.fidelity_threshold,
        utility_threshold=args.utility_threshold,
        run_auditor_step=not args.no_auditor,
        output_dir=args.output_dir,
        forcar_saida=args.forcar_saida,
        rng_seed=args.rng_seed,
    )
    resultado = run_synap(config)

    if resultado.pausado_por_homeostasia:
        print("⏸️  GERAÇÃO PAUSADA — divergência distribucional crítica detectada.")
    elif resultado.resultado_compilador and resultado.resultado_compilador.liberado:
        print(f"✅ CSV libertado: {resultado.resultado_compilador.csv_path}")
        print(f"   Relatório de Confiança (M7): {resultado.resultado_compilador.report_path}")
        if resultado.resultado_auditor:
            veredito = "APROVADO" if resultado.resultado_auditor.aprovado_global else "BLOQUEADO"
            print(f"   Relatório de Prontidão (M8): {resultado.resultado_auditor.report_path} [{veredito}]")
    else:
        print("🚫 SAÍDA BLOQUEADA pelo Compilador (Módulo 7).")
        if resultado.resultado_compilador:
            print(f"   Diagnóstico: {resultado.resultado_compilador.diagnostico}")

    for w in resultado.warnings:
        print(f"   ⚠ {w}")


if __name__ == "__main__":
    main()
