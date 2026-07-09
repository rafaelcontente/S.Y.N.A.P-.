"""Corre o pipeline completo do S.Y.N.A.P. sobre um dataset de referência
com processo gerador CONHECIDO, e recolhe todas as métricas necessárias
para o Relatório de Validação e Fiabilidade.

Metodologia: em vez de comparar contra um dataset real arbitrário (cujo
processo gerador verdadeiro é desconhecido — não permite distinguir
"parece parecido por coincidência" de "recuperou a estrutura real"),
construímos um dataset de referência com relações causais e estatísticas
CONHECIDAS À PARTIDA. Isto é a prática padrão em estudos de simulação
para validar pipelines geradores: só um processo com verdade fundamental
conhecida permite verificar se o sistema recuperou a estrutura real, e
não apenas uma amostra que "parece" real por acaso.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synap import auditor
from synap import compilador
from synap import hipocampo
from synap import hipotalamo
from synap import homeostase
from synap import neocortex
from synap import nucleo

RNG_SEED = 20260708
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(exist_ok=True, parents=True)

t0 = time.time()
log = []
def stamp(msg):
    elapsed = time.time() - t0
    line = f"[{elapsed:7.1f}s] {msg}"
    print(line)
    log.append(line)

# ============================================================
# 1. DATASET DE REFERÊNCIA — processo gerador conhecido
# ============================================================
stamp("A construir o dataset de referência (processo gerador conhecido)...")
rng = np.random.default_rng(RNG_SEED)
N_REAL = 2200

TRUE_CORR = {
    ("idade", "rendimento"): 0.42,
    ("rendimento", "divida"): -0.35,
    ("rendimento", "score_credito"): 0.55,
    ("divida", "score_credito"): -0.50,
}

z_idade = rng.normal(0, 1, N_REAL)
z_rendimento = TRUE_CORR[("idade", "rendimento")] * z_idade + np.sqrt(1 - TRUE_CORR[("idade", "rendimento")] ** 2) * rng.normal(0, 1, N_REAL)
z_divida = TRUE_CORR[("rendimento", "divida")] * z_rendimento + np.sqrt(1 - TRUE_CORR[("rendimento", "divida")] ** 2) * rng.normal(0, 1, N_REAL)

# score_credito depende de rendimento E divida (regressão múltipla verdadeira)
r_rs, r_ds = TRUE_CORR[("rendimento", "score_credito")], TRUE_CORR[("divida", "score_credito")]
r_rd = TRUE_CORR[("rendimento", "divida")]
Rxx = np.array([[1, r_rd], [r_rd, 1]])
rxy = np.array([r_rs, r_ds])
beta = np.linalg.solve(Rxx, rxy)
var_explicada = rxy @ beta
z_score = beta[0] * z_rendimento + beta[1] * z_divida + np.sqrt(max(1 - var_explicada, 1e-6)) * rng.normal(0, 1, N_REAL)

idade = np.clip(40 + 13 * z_idade, 18, 80)
rendimento = np.clip(2800 + 1300 * z_rendimento, 300, None)
divida = np.clip(8000 + 6000 * z_divida, 0, None)
score_credito = np.clip(650 + 90 * z_score, 300, 850)
risco = np.where(score_credito < 580, "alto", np.where(score_credito < 700, "medio", "baixo"))

real_df = pd.DataFrame({
    "idade": idade, "rendimento": rendimento, "divida": divida,
    "score_credito": score_credito, "risco": risco,
})

# split honesto: ancora (usada para construir o Contrato + M8 training_reference),
# holdout (nunca usada para nada além de M7/M8 — a "vida real" nunca vista)
shuffled = real_df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
cut = int(len(shuffled) * 0.68)
anchor_df = shuffled.iloc[:cut].reset_index(drop=True)
holdout_df = shuffled.iloc[cut:].reset_index(drop=True)
stamp(f"Dataset de referência: {len(real_df)} linhas ({len(anchor_df)} âncora, {len(holdout_df)} holdout nunca visto)")

# ============================================================
# 2. ESQUEMA + MÓDULO 1
# ============================================================
schema = {
    "colunas": {
        "idade": {"tipo": "numerico", "minimo": 18, "maximo": 80},
        "rendimento": {"tipo": "numerico", "minimo": 0},
        "divida": {"tipo": "numerico", "minimo": 0},
        "score_credito": {"tipo": "numerico", "minimo": 300, "maximo": 850},
        "risco": {"tipo": "categorico", "categorias": ["baixo", "medio", "alto"]},
    },
    "regras": ["rendimento > 0", "divida >= 0"],
}
schema_path = OUTDIR / "schema.yaml"
schema_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
anchor_path = OUTDIR / "anchor.csv"
anchor_df.to_csv(anchor_path, index=False)

stamp("MÓDULO 1 — a construir o Contrato Estatístico...")
contract = neocortex.build_contract(schema_path=schema_path, rules=schema["regras"], anchor_csv=anchor_path)
stamp(f"Contrato construído: fonte={contract.source}, {len(contract.correlations)} correlações, n={contract.sample_size}")

# ============================================================
# 3. MÓDULO 2 — DAG + Semente
# ============================================================
stamp("MÓDULO 2 — a validar a DAG e a gerar a semente...")
# DAG fornecida pelo utilizador — o modo de uso primário do Módulo 2
# ("recebe a DAG fornecida pelo utilizador"). Sem isto, colunas
# categóricas ficam como nós isolados (o Contrato só regista
# correlações entre numéricas) — ver Secção de Limitações no relatório
# para a quantificação honesta desse cenário alternativo.
user_dag_edges = [
    ("idade", "rendimento"),
    ("rendimento", "divida"),
    ("rendimento", "score_credito"),
    ("divida", "score_credito"),
    ("score_credito", "risco"),
]
hip_out = hipocampo.run_hipocampo(contract, user_dag_edges=user_dag_edges, n_rows=5000, rng_seed=RNG_SEED)
dag = hip_out.dag
seed = hip_out.seed
stamp(f"DAG: {len(dag.arestas)} arestas — {[(e.origem, e.destino) for e in dag.arestas]}")
stamp(f"Semente: {len(seed)} linhas, erro médio de correlação={hip_out.relatorio.erro_medio_absoluto:.4f}, ajustes={len(hip_out.ajustes)}")

# ============================================================
# 4. LOOP COGNITIVO — Núcleo (M3+4+5) + Hipotálamo (M4) + Homeostase (M6)
# ============================================================
N_BATCHES = 5
BATCH_SIZE = 900
TARGET_ROWS = N_BATCHES * BATCH_SIZE

estado_hipotalamo = hipotalamo.HipotalamoState(contract)
estado_homeostase = homeostase.HomeostaseState(contract)
memoria = seed
pesos = np.ones(len(seed))

lotes_gerados = []
rejeicoes_agregadas = {}
artefactos_por_lote = []
homeostase_por_lote = []
derivas_causais = []
regras_ilp = []

for i in range(N_BATCHES):
    stamp(f"Lote {i+1}/{N_BATCHES} — Núcleo (M3+M4+M5)...")
    saida_nucleo = nucleo.run_nucleo(
        seed=memoria, contract=contract, n_rows=BATCH_SIZE, dag=dag,
        initial_weights=pesos, rng_seed=RNG_SEED + i,
    )
    lotes_gerados.append(saida_nucleo.lote)
    for motivo, n in saida_nucleo.relatorio.rejeicoes_por_motivo.items():
        rejeicoes_agregadas[motivo] = rejeicoes_agregadas.get(motivo, 0) + n
    artefactos_por_lote.append(saida_nucleo.relatorio.artefactos_detectados)

    saida_hipotalamo = hipotalamo.run_hipotalamo(
        saida_nucleo.lote, contract, dag, state=estado_hipotalamo,
        checkpoint_size=BATCH_SIZE, rng_seed=RNG_SEED,
    )
    if saida_hipotalamo.monitorizacao_causal:
        derivas_causais.append({
            "lote": i + 1,
            "n_linhas": saida_hipotalamo.monitorizacao_causal.n_linhas_analisadas,
            "derivas": len(saida_hipotalamo.monitorizacao_causal.derivas_detectadas),
        })
    regras_ilp.extend(saida_hipotalamo.novas_regras)

    saida_homeostase = homeostase.run_controlador_homeostasia(
        saida_nucleo.lote, contract, memory=saida_nucleo.memoria_final, state=estado_homeostase,
    )
    homeostase_por_lote.append({
        "lote": i + 1,
        "divergencia_global": saida_homeostase.relatorio.divergencia_global_suavizada,
        "estado": saida_homeostase.relatorio.estado_global.value,
        "colunas": {c.coluna: c.divergencia_suavizada for c in saida_homeostase.relatorio.divergencias_colunas},
        "entropias": {e.coluna: e.entropia_normalizada for e in saida_homeostase.relatorio.entropias},
    })

    memoria = saida_nucleo.memoria_final
    pesos = np.array(saida_nucleo.pesos_finais) * np.array(saida_homeostase.pesos_multiplicador)
    stamp(f"  -> {len(saida_nucleo.lote)} linhas aprovadas; homeostasia={saida_homeostase.relatorio.estado_global.value}; artefactos={saida_nucleo.relatorio.artefactos_detectados}")

dataset_final = pd.concat(lotes_gerados, ignore_index=True)
stamp(f"Dataset final gerado: {len(dataset_final)} linhas")

# ============================================================
# 5. MÓDULO 7 — Compilador (ASP final + Modelo Gémeo + Índice de Fidelidade)
# ============================================================
stamp("MÓDULO 7 — validação ASP final + Modelo Gémeo...")
resultado_compilador = compilador.run_compilador(
    dataset_final, contract, target_column="risco", real_holdout=holdout_df,
    output_dir=OUTDIR / "m7_output", rng_seed=RNG_SEED,
)
stamp(f"Compilador: ASP aprovado={resultado_compilador.validacao_asp.aprovado}, "
      f"Índice de Fidelidade={resultado_compilador.fidelidade.indice_fidelidade:.4f}, "
      f"liberado={resultado_compilador.liberado}")

# ============================================================
# 6. MÓDULO 8 — Auditor (utilidade + privacidade)
# ============================================================
stamp("MÓDULO 8 — auditoria de utilidade e privacidade...")
resultado_auditor = auditor.run_auditor(
    dataset_final, contract, real_holdout=holdout_df, training_reference=anchor_df,
    target_columns=["risco", "score_credito"], output_dir=OUTDIR / "m8_output", rng_seed=RNG_SEED,
)
stamp(f"Auditor: utilidade aprovada={resultado_auditor.utilidade.aprovado} "
      f"(retenção média={resultado_auditor.utilidade.retencao_media:.3f}), "
      f"privacidade aprovada={resultado_auditor.privacidade.aprovado}")

stamp("Verificação honesta: efeito de fornecer (ou não) a DAG na ligação categórica-numérica...")
hip_out_sem_dag = hipocampo.run_hipocampo(contract, n_rows=5000, rng_seed=RNG_SEED)
seed_sem_dag = hip_out_sem_dag.seed

def _associacao_categoria_numerica(df, cat_col, num_col):
    """Eta-quadrado simples: variância entre grupos / variância total."""
    total_var = df[num_col].var()
    if total_var == 0:
        return 0.0
    medias_grupo = df.groupby(cat_col)[num_col].mean()
    contagens = df.groupby(cat_col)[num_col].count()
    media_global = df[num_col].mean()
    var_entre_grupos = ((medias_grupo - media_global) ** 2 * contagens).sum() / len(df)
    return float(var_entre_grupos / total_var)

associacao_com_dag = _associacao_categoria_numerica(seed, "risco", "score_credito")
associacao_sem_dag = _associacao_categoria_numerica(seed_sem_dag, "risco", "score_credito")
associacao_real = _associacao_categoria_numerica(real_df, "risco", "score_credito")
stamp(f"Associação risco~score_credito (eta²): real={associacao_real:.3f}, com DAG={associacao_com_dag:.3f}, sem DAG={associacao_sem_dag:.3f}")

# ============================================================
# 7. TESTES ESTATÍSTICOS ADICIONAIS DE REALISMO
# ============================================================
stamp("Testes adicionais: Kolmogorov-Smirnov, Qui-quadrado, recuperação da estrutura verdadeira...")
numeric_cols = ["idade", "rendimento", "divida", "score_credito"]

ks_results = {}
for col in numeric_cols:
    stat, p = scipy_stats.ks_2samp(dataset_final[col], holdout_df[col])
    ks_results[col] = {"statistic": float(stat), "p_value": float(p), "indistinguivel": bool(p > 0.05)}

# qui-quadrado para a categórica
cats = ["baixo", "medio", "alto"]
obs_synth = dataset_final["risco"].value_counts().reindex(cats, fill_value=0).to_numpy()
obs_real = holdout_df["risco"].value_counts().reindex(cats, fill_value=0).to_numpy()
expected = obs_real / obs_real.sum() * obs_synth.sum()
chi2_stat, chi2_p = scipy_stats.chisquare(obs_synth, f_exp=expected)

# recuperação da estrutura de correlação verdadeira (não só vs. holdout, vs. o PROCESSO GERADOR)
recovered_corr = {}
for (a, b), true_val in TRUE_CORR.items():
    synth_val = float(dataset_final[a].corr(dataset_final[b]))
    real_val = float(real_df[a].corr(real_df[b]))
    recovered_corr[f"{a}-{b}"] = {"verdadeiro": true_val, "real_amostrado": real_val, "sintetico": synth_val,
                                   "erro_absoluto": abs(synth_val - true_val)}

stamp(f"KS tests: {sum(1 for v in ks_results.values() if v['indistinguivel'])}/{len(ks_results)} colunas indistinguíveis (p>0.05)")
stamp(f"Qui-quadrado risco: p={chi2_p:.4f}")

# ============================================================
# 8. GUARDAR TUDO PARA A FASE DE GRÁFICOS/PDF
# ============================================================
stamp("A guardar resultados...")

results = {
    "n_real": len(real_df), "n_anchor": len(anchor_df), "n_holdout": len(holdout_df),
    "n_seed": len(seed), "n_generated": len(dataset_final),
    "true_corr": {f"{a}-{b}": v for (a, b), v in TRUE_CORR.items()},
    "dag_edges": [(e.origem, e.destino) for e in dag.arestas],
    "seed_report": {
        "erro_medio_absoluto": hip_out.relatorio.erro_medio_absoluto,
        "erro_maximo_absoluto": hip_out.relatorio.erro_maximo_absoluto,
        "n_ajustes": len(hip_out.ajustes),
    },
    "rejeicoes_agregadas": rejeicoes_agregadas,
    "artefactos_por_lote": artefactos_por_lote,
    "homeostase_por_lote": homeostase_por_lote,
    "derivas_causais": derivas_causais,
    "n_regras_ilp": len(regras_ilp),
    "compilador": {
        "asp_total": resultado_compilador.validacao_asp.total_linhas,
        "asp_aprovadas": resultado_compilador.validacao_asp.aprovadas,
        "asp_rejeitadas": resultado_compilador.validacao_asp.rejeitadas,
        "coluna_alvo": resultado_compilador.fidelidade.coluna_alvo,
        "metrica_nome": resultado_compilador.fidelidade.metrica_sintetica.metrica_nome,
        "valor_sintetico": resultado_compilador.fidelidade.metrica_sintetica.valor,
        "valor_real": resultado_compilador.fidelidade.metrica_real.valor,
        "indice_fidelidade": resultado_compilador.fidelidade.indice_fidelidade,
        "liberado": resultado_compilador.liberado,
        "diagnostico": resultado_compilador.diagnostico,
    },
    "auditor": {
        "tarefas": [
            {"coluna_alvo": t.coluna_alvo, "metrica_nome": t.metrica_nome, "tstr": t.valor_tstr,
             "trtr": t.valor_trtr, "retencao": t.retencao_utilidade}
            for t in resultado_auditor.utilidade.tarefas
        ],
        "retencao_media": resultado_auditor.utilidade.retencao_media,
        "utilidade_aprovada": resultado_auditor.utilidade.aprovado,
        "dcr_sintetico_real": resultado_auditor.privacidade.dcr.dcr_sintetico_real_mediana,
        "dcr_real_real": resultado_auditor.privacidade.dcr.dcr_real_real_mediana,
        "dcr_razao": resultado_auditor.privacidade.dcr.razao,
        "dcr_linhas_suspeitas": resultado_auditor.privacidade.dcr.linhas_suspeitas,
        "nndr_sintetico": resultado_auditor.privacidade.nndr.nndr_sintetico_mediana,
        "nndr_real": resultado_auditor.privacidade.nndr.nndr_real_mediana,
        "correspondencias_exactas": resultado_auditor.privacidade.correspondencia_exacta.n_correspondencias_exactas,
        "mia_avaliavel": resultado_auditor.privacidade.mia.avaliavel,
        "mia_auc": resultado_auditor.privacidade.mia.auc,
        "mia_seguro": resultado_auditor.privacidade.mia.seguro,
        "privacidade_aprovada": resultado_auditor.privacidade.aprovado,
        "privacidade_motivo": resultado_auditor.privacidade.motivo,
    },
    "ks_results": ks_results,
    "chi2": {"statistic": float(chi2_stat), "p_value": float(chi2_p)},
    "recovered_corr": recovered_corr,
    "associacao_categoria_numerica": {
        "real": associacao_real, "com_dag": associacao_com_dag, "sem_dag": associacao_sem_dag,
    },
    "log": log,
}

with open(OUTDIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

# guardar datasets para os gráficos
real_df.to_parquet(OUTDIR / "real_df.parquet")
holdout_df.to_parquet(OUTDIR / "holdout_df.parquet")
anchor_df.to_parquet(OUTDIR / "anchor_df.parquet")
dataset_final.to_parquet(OUTDIR / "synthetic_df.parquet")

with open(OUTDIR / "contract.pkl", "wb") as f:
    pickle.dump(contract, f)

stamp("Concluído. Resultados guardados em validation/results.json")
