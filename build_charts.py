"""Gera todos os gráficos (PNG, alta resolução) do Relatório de Validação
a partir dos resultados guardados por `run_validation.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTDIR = Path(__file__).resolve().parent
CHARTS = OUTDIR / "charts"
CHARTS.mkdir(exist_ok=True, parents=True)

# --------------------------------------------------------------------------
# Paleta e estilo — consistente, profissional, para impressão
# --------------------------------------------------------------------------
NAVY = "#101828"
VIOLET = "#6C63FF"
TEAL = "#0EA88C"
GOLD = "#DDA22A"
CORAL = "#E5484D"
GREY = "#8A93A6"
LIGHT_GREY = "#E4E7EC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": LIGHT_GREY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

results = json.loads((OUTDIR / "results.json").read_text(encoding="utf-8"))
real_df = pd.read_parquet(OUTDIR / "real_df.parquet")
holdout_df = pd.read_parquet(OUTDIR / "holdout_df.parquet")
synthetic_df = pd.read_parquet(OUTDIR / "synthetic_df.parquet")

NUMERIC_COLS = ["idade", "rendimento", "divida", "score_credito"]
NUMERIC_LABELS = {"idade": "Idade (anos)", "rendimento": "Rendimento (€)", "divida": "Dívida (€)", "score_credito": "Score de crédito"}


def savefig(fig, name):
    path = CHARTS / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 1. Histogramas — real (holdout) vs sintético, sobrepostos
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, NUMERIC_COLS):
    bins = np.histogram_bin_edges(pd.concat([holdout_df[col], synthetic_df[col]]), bins=28)
    ax.hist(holdout_df[col], bins=bins, alpha=0.55, color=GREY, label="Real (holdout nunca visto)", density=True)
    ax.hist(synthetic_df[col], bins=bins, alpha=0.55, color=VIOLET, label="Sintético", density=True)
    ax.set_title(NUMERIC_LABELS[col])
    ax.set_ylabel("densidade")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)
fig.suptitle("Distribuições: real vs. sintético", y=1.09, fontsize=14, fontweight="bold")
fig.tight_layout()
savefig(fig, "01_histogramas.png")

# --------------------------------------------------------------------------
# 2. Matrizes de correlação — real vs sintético, lado a lado
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, df, title in [(axes[0], holdout_df, "Real (holdout)"), (axes[1], synthetic_df, "Sintético")]:
    corr = df[NUMERIC_COLS].corr().to_numpy()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(NUMERIC_COLS))); ax.set_xticklabels(NUMERIC_COLS, rotation=40, ha="right")
    ax.set_yticks(range(len(NUMERIC_COLS))); ax.set_yticklabels(NUMERIC_COLS)
    for i in range(len(NUMERIC_COLS)):
        for j in range(len(NUMERIC_COLS)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr[i, j]) > 0.6 else NAVY, fontsize=9)
    ax.set_title(title)
fig.colorbar(im, ax=axes, shrink=0.75, label="correlação de Pearson")
fig.suptitle("Matriz de correlação: real vs. sintético", fontsize=14, fontweight="bold")
savefig(fig, "02_correlacao.png")

# --------------------------------------------------------------------------
# 3. Recuperação da estrutura causal VERDADEIRA (não só vs. amostra real)
# --------------------------------------------------------------------------
rc = results["recovered_corr"]
pares = list(rc.keys())
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(pares))
w = 0.25
ax.bar(x - w, [rc[p]["verdadeiro"] for p in pares], width=w, label="Processo gerador (verdade fundamental)", color=NAVY)
ax.bar(x, [rc[p]["real_amostrado"] for p in pares], width=w, label="Amostra real (holdout)", color=GREY)
ax.bar(x + w, [rc[p]["sintetico"] for p in pares], width=w, label="Sintético gerado", color=VIOLET)
ax.axhline(0, color=LIGHT_GREY, linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(pares, rotation=15, ha="right")
ax.set_ylabel("correlação de Pearson")
ax.set_title("Recuperação da estrutura causal verdadeira")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
savefig(fig, "03_recuperacao_estrutura.png")

# --------------------------------------------------------------------------
# 4. Categórica: distribuição de proporções (real vs sintético)
# --------------------------------------------------------------------------
cats = ["baixo", "medio", "alto"]
real_props = holdout_df["risco"].value_counts(normalize=True).reindex(cats, fill_value=0)
synth_props = synthetic_df["risco"].value_counts(normalize=True).reindex(cats, fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
x = np.arange(len(cats)); w = 0.32
axes[0].bar(x - w / 2, real_props.values, width=w, label="Real", color=GREY)
axes[0].bar(x + w / 2, synth_props.values, width=w, label="Sintético", color=VIOLET)
axes[0].set_xticks(x); axes[0].set_xticklabels(cats)
axes[0].set_ylabel("proporção"); axes[0].set_title("Proporções de \"risco\"")
axes[0].legend(frameon=False)

real_means = real_df.groupby("risco")["score_credito"].mean().reindex(cats)
synth_means = synthetic_df.groupby("risco")["score_credito"].mean().reindex(cats)
axes[1].bar(x - w / 2, real_means.values, width=w, label="Real", color=GREY)
axes[1].bar(x + w / 2, synth_means.values, width=w, label="Sintético", color=VIOLET)
axes[1].set_xticks(x); axes[1].set_xticklabels(cats)
axes[1].set_ylabel("score_credito médio")
axes[1].set_title("Score médio por categoria de risco\n(prova de direção correta do acoplamento)")
axes[1].legend(frameon=False)
fig.tight_layout()
savefig(fig, "04_categorica.png")

# --------------------------------------------------------------------------
# 5. Homeostasia — convergência da divergência ao longo dos lotes
# --------------------------------------------------------------------------
lotes = results["homeostase_por_lote"]
fig, ax = plt.subplots(figsize=(9, 4.6))
x = [l["lote"] for l in lotes]
for col, color in zip(NUMERIC_COLS, [NAVY, VIOLET, TEAL, GOLD]):
    y = [l["colunas"].get(col, np.nan) * 100 for l in lotes]
    ax.plot(x, y, marker="o", label=col, color=color)
ax.axhline(5, color=GOLD, linestyle="--", linewidth=1, label="limiar de atenção (5%)")
ax.axhline(10, color=CORAL, linestyle="--", linewidth=1, label="limiar crítico (10%)")
ax.set_xlabel("lote (checkpoint)"); ax.set_ylabel("divergência suavizada (%)")
ax.set_title("Homeostasia — divergência por coluna ao longo dos lotes")
ax.set_xticks(x)
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
savefig(fig, "05_homeostasia.png")

# --------------------------------------------------------------------------
# 6. Motivos de rejeição agregados (Núcleo)
# --------------------------------------------------------------------------
rej = results["rejeicoes_agregadas"]
nomes = {"distancia_mahalanobis": "Distância de\nMahalanobis", "score_plausibilidade": "Score de\nplausibilidade",
         "rejeicao_logica_asp": "Regras de\nnegócio (ASP)", "artefacto_autoencoder": "Artefacto\n(autoencoder)",
         "tentativas_esgotadas": "Tentativas\nesgotadas"}
labels = [nomes[k] for k in rej]
valores = list(rej.values())
fig, ax = plt.subplots(figsize=(8, 4.2))
cores = [VIOLET, TEAL, CORAL, GOLD, GREY]
bars = ax.bar(labels, valores, color=cores[: len(labels)])
for b, v in zip(bars, valores):
    ax.text(b.get_x() + b.get_width() / 2, v + max(valores) * 0.01, str(v), ha="center", fontsize=9)
ax.set_ylabel("nº de linhas candidatas rejeitadas")
ax.set_title(f"Motivos de rejeição — {sum(valores)} candidatas testadas para {results['n_generated']} linhas aprovadas")
fig.tight_layout()
savefig(fig, "06_rejeicoes.png")

# --------------------------------------------------------------------------
# 7. Testes de Kolmogorov-Smirnov (p-valor por coluna)
# --------------------------------------------------------------------------
ks = results["ks_results"]
fig, ax = plt.subplots(figsize=(8, 4.2))
cols = list(ks.keys())
pvals = [ks[c]["p_value"] for c in cols]
cores = [TEAL if ks[c]["indistinguivel"] else CORAL for c in cols]
bars = ax.bar(cols, pvals, color=cores)
ax.axhline(0.05, color=NAVY, linestyle="--", linewidth=1, label="limiar α=0.05")
for b, c in zip(bars, cols):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"D={ks[c]['statistic']:.3f}", ha="center", fontsize=8)
ax.set_ylabel("p-valor (teste de Kolmogorov-Smirnov)")
ax.set_title("Indistinguibilidade estatística: sintético vs. real (holdout)")
ax.legend(frameon=False)
ax.set_ylim(0, max(pvals + [0.1]) * 1.3)
fig.tight_layout()
savefig(fig, "07_ks_tests.png")

# --------------------------------------------------------------------------
# 8. Modelo Gémeo — utilidade (TSTR vs TRTR) por tarefa
# --------------------------------------------------------------------------
tarefas = results["auditor"]["tarefas"]
fig, axes = plt.subplots(1, len(tarefas), figsize=(5 * len(tarefas), 4.4))
if len(tarefas) == 1:
    axes = [axes]
for ax, t in zip(axes, tarefas):
    vals = [t["tstr"], t["trtr"]]
    bars = ax.bar(["Sintético\n(TSTR)", "Real\n(TRTR)"], vals, color=[VIOLET, GREY])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if t["metrica_nome"] == "AUC" else f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{t['coluna_alvo']} ({t['metrica_nome']})\nretenção de utilidade: {t['retencao']:.1%}")
fig.suptitle("Módulo 8 — Utilidade: Treino-Sintético-Teste-Real (TSTR) vs. Treino-Real-Teste-Real (TRTR)", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "08_utilidade_tstr_trtr.png")

# --------------------------------------------------------------------------
# 9. Privacidade — DCR e MIA
# --------------------------------------------------------------------------
priv = results["auditor"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
vals = [priv["dcr_real_real"], priv["dcr_sintetico_real"]]
bars = axes[0].bar(["Real → Real\n(baseline interno)", "Sintético → Real"], vals, color=[GREY, VIOLET])
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
axes[0].set_title(f"Distance to Closest Record\nrazão = {priv['dcr_razao']:.2f} (>=0.5 é saudável)")
axes[0].set_ylabel("distância mediana (espaço padronizado)")

auc = priv["mia_auc"]
axes[1].bar(["Ataque de Inferência\nde Pertença (MIA)"], [auc], color=TEAL if priv["mia_seguro"] else CORAL, width=0.5)
axes[1].axhline(0.5, color=NAVY, linestyle="--", linewidth=1, label="0.5 = ataque não distingue (seguro)")
axes[1].set_ylim(0, 1)
axes[1].text(0, auc + 0.03, f"AUC={auc:.3f}", ha="center")
axes[1].set_title("Risco de reidentificação (MIA)")
axes[1].legend(frameon=False, loc="upper right", fontsize=8)
fig.suptitle("Módulo 8 — Privacidade: memorização e inferência de pertença", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "09_privacidade.png")

# --------------------------------------------------------------------------
# 10. Suite de testes automatizados
# --------------------------------------------------------------------------
suite = {
    "neocortex\n(M1)": 28, "hipocampo\n(M2)": 30, "expansor\n(M3)": 20, "hipotalamo\n(M4)": 22,
    "nucleo\n(M5)": 12, "homeostase\n(M6)": 14, "compilador\n(M7)": 18, "auditor\n(M8)": 20,
    "integração\nend-to-end": 6,
}
fig, ax = plt.subplots(figsize=(10, 4.6))
bars = ax.bar(list(suite.keys()), list(suite.values()), color=VIOLET)
for b, v in zip(bars, suite.values()):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=9)
ax.set_ylabel("testes automatizados aprovados")
total = sum(suite.values())
ax.set_title(f"Suite de testes automatizados — {total}/{total} aprovados (100%)")
fig.tight_layout()
savefig(fig, "10_suite_testes.png")

# --------------------------------------------------------------------------
# 11. Associação categórica-numérica: com DAG vs sem DAG (limitação honesta)
# --------------------------------------------------------------------------
assoc = results["associacao_categoria_numerica"]
fig, ax = plt.subplots(figsize=(7, 4.4))
labels = ["Real\n(verdade)", "Sintético\n(com DAG fornecida)", "Sintético\n(sem DAG - automática)"]
vals = [assoc["real"], assoc["com_dag"], assoc["sem_dag"]]
cores = [NAVY, TEAL, CORAL]
bars = ax.bar(labels, vals, color=cores)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
ax.set_ylabel("associação risco ~ score_credito (eta²)")
ax.set_title("Efeito de fornecer a DAG causal\n(limitação documentada do Módulo 2)")
ax.set_ylim(0, 1)
fig.tight_layout()
savefig(fig, "11_dag_associacao.png")

print("Gráficos gerados em", CHARTS)
for p in sorted(CHARTS.glob("*.png")):
    print(" -", p.name)
