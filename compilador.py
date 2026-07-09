"""Neocórtex Motor, Validador por Proxy (Modelo Gémeo) e Compilador de Confiança.

Módulo 7 do S.Y.N.A.P. — o "guardião da porta" final. Revalida 100% do
dataset final contra o ASP, treina um Modelo Gémeo (Regressão
Logística/Linear) para provar empiricamente o realismo do dataset,
calcula o Índice de Fidelidade comparando com um holdout real (se
disponível) e, só se este for suficientemente alto, compila o CSV
final e o Relatório de Confiança em PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from synap.hipotalamo.hipotalamo import validate_batch
from synap.neocortex.models import CategoricalStatistics, ContinuousStatistics, StatisticalContract

from .exceptions import CompiladorValidationError
from .models import (
    ASPFinalValidationResult,
    CompilerOutput,
    FidelityAssessment,
    TargetType,
    TwinModelMetrics,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

DEFAULT_FIDELITY_THRESHOLD: float = 0.90
DEFAULT_BLOCK_THRESHOLD_FOR_REPORTING: float = 0.80
"""Limiar de referência usado apenas em mensagens de diagnóstico
("Índice < 0.80" citado no enunciado como caso de bloqueio evidente)."""

DEFAULT_TEST_SIZE: float = 0.2
MIN_ROWS_FOR_SPLIT: int = 50
"""Mínimo de linhas para um split 80/20 com sinal estatístico razoável."""

TOP_N_ARTIFACTS: int = 10


# --------------------------------------------------------------------------
# Frente 1 — Validação ASP exaustiva final
# --------------------------------------------------------------------------


def run_final_asp_validation(
    dataset: pd.DataFrame, contract: StatisticalContract
) -> ASPFinalValidationResult:
    """Revalida exaustivamente 100% do dataset final contra as regras do Contrato.

    Camada de segurança redundante com o Módulo 4 — em condições
    normais não deve encontrar nenhuma violação, mas o compilador nunca
    assume isso sem verificar.
    """
    relatorio, _ = validate_batch(dataset, contract.rules)
    return ASPFinalValidationResult(
        total_linhas=relatorio.total_linhas,
        aprovadas=relatorio.aprovadas,
        rejeitadas=relatorio.rejeitadas,
        aprovado=relatorio.rejeitadas == 0,
    )


# --------------------------------------------------------------------------
# Seleção da coluna-alvo e codificação de features
# --------------------------------------------------------------------------


def select_target_column(
    contract: StatisticalContract, target_column: str | None = None
) -> str:
    """Seleciona a coluna-alvo do Modelo Gémeo.

    Se `target_column` não for fornecida, escolhe a coluna numérica de
    maior variância (desvio² do Contrato); se não existir nenhuma
    coluna numérica, usa a primeira coluna categórica declarada.

    Raises:
        CompiladorValidationError: Se `target_column` for fornecida mas
            não existir no Contrato.
    """
    if target_column is not None:
        if target_column not in contract.column_stats:
            raise CompiladorValidationError(
                f"coluna-alvo '{target_column}' não existe no Contrato Estatístico"
            )
        return target_column

    numeric_cols = [
        (nome, stat.desvio**2)
        for nome, stat in contract.column_stats.items()
        if isinstance(stat, ContinuousStatistics)
    ]
    if numeric_cols:
        return max(numeric_cols, key=lambda item: item[1])[0]

    categorical_cols = [
        nome for nome, stat in contract.column_stats.items() if isinstance(stat, CategoricalStatistics)
    ]
    if categorical_cols:
        return categorical_cols[0]

    raise CompiladorValidationError("o Contrato não declara nenhuma coluna utilizável como alvo")


def _encode_features(
    df: pd.DataFrame, contract: StatisticalContract, target_column: str
) -> np.ndarray:
    """Codifica todas as colunas exceto `target_column` em features numéricas.

    Numéricas são padronizadas (z-score) com média/desvio do Contrato;
    categóricas usam o mapa ordinal das categorias declaradas — a mesma
    convenção de codificação usada nos Módulos 3 e 5.
    """
    colunas_features = [c for c in df.columns if c != target_column]
    partes = []
    for coluna in colunas_features:
        stat = contract.column_stats[coluna]
        if isinstance(stat, ContinuousStatistics):
            desvio = max(stat.desvio, 1e-9)
            partes.append(((df[coluna].to_numpy(dtype=float) - stat.media) / desvio).reshape(-1, 1))
        else:
            encoder = {categoria: i for i, categoria in enumerate(stat.frequencias)}
            codigos = df[coluna].map(lambda v, enc=encoder: enc.get(v, -1)).to_numpy(dtype=float)
            partes.append(codigos.reshape(-1, 1))
    return np.hstack(partes) if partes else np.zeros((len(df), 0))


# --------------------------------------------------------------------------
# Frente 2/3 — Modelo Gémeo e validação cruzada interna (80/20)
# --------------------------------------------------------------------------


def train_twin_model(
    df: pd.DataFrame,
    contract: StatisticalContract,
    target_column: str,
    test_size: float = DEFAULT_TEST_SIZE,
    rng_seed: int | None = None,
) -> TwinModelMetrics:
    """Treina e avalia o Modelo Gémeo num único split 80/20.

    Usa Regressão Logística (AUC) se a coluna-alvo for categórica, ou
    Regressão Linear (RMSE) se for numérica — modelos leves e
    determinísticos, deliberadamente mais simples que XGBoost, para um
    validador cuja função é atestar o realismo estatístico básico do
    dataset, não maximizar performance preditiva.

    Raises:
        CompiladorValidationError: Se `df` tiver menos de `MIN_ROWS_FOR_SPLIT` linhas.
    """
    if len(df) < MIN_ROWS_FOR_SPLIT:
        raise CompiladorValidationError(
            f"dataset com {len(df)} linha(s) é insuficiente para um split "
            f"80/20 com sinal estatístico razoável (mínimo: {MIN_ROWS_FOR_SPLIT})"
        )

    x = _encode_features(df, contract, target_column)
    y = df[target_column].to_numpy()
    is_categorico = isinstance(contract.column_stats[target_column], CategoricalStatistics)

    estratificar = y if is_categorico else None
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=test_size, random_state=rng_seed, stratify=estratificar
    )

    if is_categorico:
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(x_treino, y_treino)
        proba = modelo.predict_proba(x_teste)
        if len(modelo.classes_) == 2:
            valor = float(roc_auc_score(y_teste, proba[:, 1]))
        else:
            valor = float(roc_auc_score(y_teste, proba, multi_class="ovr", average="macro"))
        return TwinModelMetrics(
            tipo_alvo=TargetType.CATEGORICO,
            metrica_nome="AUC",
            valor=valor,
            n_treino=len(x_treino),
            n_teste=len(x_teste),
        )

    modelo = LinearRegression()
    modelo.fit(x_treino, y_treino)
    previsao = modelo.predict(x_teste)
    rmse = float(np.sqrt(mean_squared_error(y_teste, previsao)))
    return TwinModelMetrics(
        tipo_alvo=TargetType.NUMERICO,
        metrica_nome="RMSE",
        valor=rmse,
        n_treino=len(x_treino),
        n_teste=len(x_teste),
    )


def compute_fidelity_index(sintetica: TwinModelMetrics, real: TwinModelMetrics) -> float:
    """Índice de Fidelidade: `1 - |métrica_sintética - métrica_real| / métrica_real`.

    A mesma fórmula de diferença relativa serve para AUC (quanto maior
    melhor) e RMSE (quanto menor melhor) — o que importa é a magnitude
    do desvio relativamente ao desempenho real, não a sua direção.
    """
    if real.valor == 0:
        return 0.0
    diferenca = abs(sintetica.valor - real.valor) / abs(real.valor)
    return float(1 - diferenca)


def run_twin_model_validation(
    dataset: pd.DataFrame,
    contract: StatisticalContract,
    target_column: str | None = None,
    real_holdout: pd.DataFrame | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    rng_seed: int | None = None,
) -> FidelityAssessment:
    """Executa a validação por proxy completa: Modelo Gémeo + Índice de Fidelidade.

    Args:
        dataset: O dataset sintético final.
        contract: O Contrato Estatístico.
        target_column: Coluna-alvo (ver :func:`select_target_column`).
        real_holdout: Amostra de dados reais para comparação, se disponível.
        test_size: Fração usada para teste no split interno 80/20.
        rng_seed: Semente aleatória, para reprodutibilidade.

    Returns:
        :class:`FidelityAssessment` com as métricas e o Índice de
        Fidelidade (`None` se `real_holdout` não for fornecido).
    """
    alvo = select_target_column(contract, target_column)
    metrica_sintetica = train_twin_model(dataset, contract, alvo, test_size, rng_seed)

    metrica_real = None
    diferenca = None
    indice = None
    if real_holdout is not None:
        metrica_real = train_twin_model(real_holdout, contract, alvo, test_size, rng_seed)
        indice = compute_fidelity_index(metrica_sintetica, metrica_real)
        diferenca = 1 - indice

    return FidelityAssessment(
        coluna_alvo=alvo,
        metrica_sintetica=metrica_sintetica,
        metrica_real=metrica_real,
        diferenca_relativa=diferenca,
        indice_fidelidade=indice,
    )


# --------------------------------------------------------------------------
# Compilação do CSV final
# --------------------------------------------------------------------------


def compile_dataset(dataset: pd.DataFrame, output_path: Path) -> Path:
    """Compila o dataset final para `.csv` (UTF-8, com cabeçalhos)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


# --------------------------------------------------------------------------
# Relatório de Confiança (PDF)
# --------------------------------------------------------------------------


def generate_confidence_report(
    dataset: pd.DataFrame,
    contract: StatisticalContract,
    fidelidade: FidelityAssessment,
    output_path: Path,
    anti_examples: list[Any] | None = None,
) -> Path:
    """Gera o Relatório de Confiança em PDF: um documento autoexplicativo
    que permite a um não-especialista perceber a decisão do compilador.

    Páginas: (1) resumo e Índice de Fidelidade; (2) histogramas das
    colunas numéricas; (3) mapa de correlação; (4) top-10 artefactos
    mais estranhos (se fornecidos pelo Módulo 5); (5) comparação do
    Modelo Gémeo sintético vs. real (se disponível).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_cols = [
        c for c, s in contract.column_stats.items() if isinstance(s, ContinuousStatistics)
    ]

    with PdfPages(output_path) as pdf:
        # --- Página 1: resumo ---
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        indice_txt = (
            f"{fidelidade.indice_fidelidade:.3f}" if fidelidade.indice_fidelidade is not None else "N/D"
        )
        linhas_resumo = [
            "Relatório de Confiança — S.Y.N.A.P.",
            "",
            f"Linhas no dataset: {len(dataset)}",
            f"Coluna-alvo do Modelo Gémeo: {fidelidade.coluna_alvo}",
            f"Métrica ({fidelidade.metrica_sintetica.metrica_nome}) sintética: "
            f"{fidelidade.metrica_sintetica.valor:.4f}",
        ]
        if fidelidade.metrica_real is not None:
            linhas_resumo.append(
                f"Métrica ({fidelidade.metrica_real.metrica_nome}) real: {fidelidade.metrica_real.valor:.4f}"
            )
        linhas_resumo.append(f"Índice de Fidelidade: {indice_txt}")
        ax.text(0.05, 0.95, "\n".join(linhas_resumo), va="top", fontsize=12, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Página 2: histogramas ---
        if numeric_cols:
            n_cols = min(2, len(numeric_cols))
            n_rows = int(np.ceil(len(numeric_cols) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.27, 11.69))
            axes_flat = np.atleast_1d(axes).flatten()
            for ax, coluna in zip(axes_flat, numeric_cols):
                ax.hist(dataset[coluna], bins=30, color="#14213D")
                ax.set_title(coluna)
            for ax in axes_flat[len(numeric_cols) :]:
                ax.axis("off")
            fig.suptitle("Distribuições (colunas numéricas)")
            pdf.savefig(fig)
            plt.close(fig)

        # --- Página 3: matriz de correlação ---
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8.27, 8.27))
            matriz = dataset[numeric_cols].corr(method="pearson")
            im = ax.imshow(matriz, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
            ax.set_yticklabels(numeric_cols)
            fig.colorbar(im, ax=ax)
            ax.set_title("Matriz de correlação")
            pdf.savefig(fig)
            plt.close(fig)

        # --- Página 4: top-N artefactos mais estranhos ---
        if anti_examples:
            top = sorted(anti_examples, key=lambda a: a.perda_reconstrucao, reverse=True)[:TOP_N_ARTIFACTS]
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            linhas = [f"Top {len(top)} artefactos mais estranhos (maior perda de reconstrução):", ""]
            for i, artefacto in enumerate(top, start=1):
                linhas.append(f"{i}. perda={artefacto.perda_reconstrucao:.4f} — {artefacto.valores}")
            ax.text(0.02, 0.98, "\n".join(linhas), va="top", fontsize=8, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)

        # --- Página 5: comparação do Modelo Gémeo ---
        if fidelidade.metrica_real is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            nomes = ["Sintético", "Real"]
            valores = [fidelidade.metrica_sintetica.valor, fidelidade.metrica_real.valor]
            ax.bar(nomes, valores, color=["#14213D", "#C9A35F"])
            ax.set_title(f"Modelo Gémeo — {fidelidade.metrica_sintetica.metrica_nome}")
            pdf.savefig(fig)
            plt.close(fig)

    return output_path


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def run_compilador(
    dataset: pd.DataFrame,
    contract: StatisticalContract,
    target_column: str | None = None,
    real_holdout: pd.DataFrame | None = None,
    fidelity_threshold: float = DEFAULT_FIDELITY_THRESHOLD,
    output_dir: Path = Path("/mnt/user-data/outputs"),
    anti_examples: list[Any] | None = None,
    forcar_saida: bool = False,
    rng_seed: int | None = None,
) -> CompilerOutput:
    """Executa o ciclo completo do "guardião da porta" (Módulo 7 do S.Y.N.A.P.).

    Args:
        dataset: O dataset sintético final consolidado.
        contract: O Contrato Estatístico do Módulo 1.
        target_column: Coluna-alvo do Modelo Gémeo (ver :func:`select_target_column`).
        real_holdout: Amostra de dados reais para o Índice de Fidelidade.
        fidelity_threshold: Índice de Fidelidade mínimo para libertar a saída.
        output_dir: Diretório onde o CSV e o PDF são escritos.
        anti_examples: Anti-exemplos do Módulo 5, para o top-10 de artefactos.
        forcar_saida: Se True, liberta a saída mesmo com Índice abaixo do
            limiar (ou sem holdout real) — ecoa a pergunta do enunciado
            "deseja... forçar a saída?", com aviso explícito.
        rng_seed: Semente aleatória, para reprodutibilidade.

    Returns:
        :class:`CompilerOutput` com a decisão final, os caminhos dos
        ficheiros gerados (se libertado) e o diagnóstico (se bloqueado).
    """
    warnings: list[str] = []
    validacao_asp = run_final_asp_validation(dataset, contract)

    if not validacao_asp.aprovado:
        return CompilerOutput(
            validacao_asp=validacao_asp,
            fidelidade=None,
            liberado=False,
            diagnostico=(
                f"Validação ASP final encontrou {validacao_asp.rejeitadas} linha(s) "
                "com violações de regras de negócio — dataset REJEITADO antes do "
                "Modelo Gémeo. Regresse ao Módulo 3/5 para regenerar."
            ),
            csv_path=None,
            report_path=None,
            warnings=warnings,
        )

    fidelidade = run_twin_model_validation(
        dataset, contract, target_column, real_holdout, rng_seed=rng_seed
    )

    diagnostico: str | None = None
    if real_holdout is None:
        liberado = forcar_saida
        if forcar_saida:
            diagnostico = (
                "sem holdout real disponível — Índice de Fidelidade não pôde ser "
                "calculado; saída FORÇADA pelo utilizador sem essa garantia"
            )
            warnings.append(diagnostico)
        else:
            diagnostico = (
                "sem holdout real disponível — Índice de Fidelidade não pôde ser "
                "calculado; saída bloqueada por omissão. Forneça um holdout real "
                "ou defina forcar_saida=True para libertar mesmo assim"
            )
    else:
        indice = fidelidade.indice_fidelidade or 0.0
        if indice >= fidelity_threshold:
            liberado = True
        elif forcar_saida:
            liberado = True
            diagnostico = (
                f"Índice de Fidelidade {indice:.3f} abaixo do limiar "
                f"({fidelity_threshold:.2f}), mas a saída foi FORÇADA pelo "
                "utilizador — os dados podem não ser suficientemente realistas"
            )
            warnings.append(diagnostico)
        else:
            liberado = False
            diagnostico = (
                f"Os dados não são realistas o suficiente. Índice de Fidelidade "
                f"{indice:.3f} < {fidelity_threshold:.2f}. Motivo: desempenho do "
                f"Modelo Gémeo ({fidelidade.metrica_sintetica.metrica_nome}="
                f"{fidelidade.metrica_sintetica.valor:.4f}) diverge do holdout real "
                f"({fidelidade.metrica_real.metrica_nome}={fidelidade.metrica_real.valor:.4f}). "
                "Deseja regenerar com ajustes ou forçar a saída?"
            )

    csv_path = None
    report_path = None
    if liberado:
        csv_path = compile_dataset(dataset, Path(output_dir) / "dataset_final.csv")
        report_path = generate_confidence_report(
            dataset, contract, fidelidade, Path(output_dir) / "relatorio_confianca.pdf", anti_examples
        )

    logger.info(
        "Compilador executado: aprovado_asp=%s, indice_fidelidade=%s, liberado=%s",
        validacao_asp.aprovado,
        fidelidade.indice_fidelidade,
        liberado,
    )

    return CompilerOutput(
        validacao_asp=validacao_asp,
        fidelidade=fidelidade,
        liberado=liberado,
        diagnostico=diagnostico,
        csv_path=str(csv_path) if csv_path else None,
        report_path=str(report_path) if report_path else None,
        warnings=warnings,
    )
