# S.Y.N.A.P.

[![CI](https://github.com/synap-project/synap/actions/workflows/ci.yml/badge.svg)](https://github.com/synap-project/synap/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/testes-170%2F170-brightgreen.svg)](./validation/Relatorio_Validacao_Fiabilidade_SYNAP.pdf)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-informational.svg)](https://github.com/astral-sh/ruff)

> For the project's history — why it exists, what it delivers, and how it compares to other synthetic data tools — see
> [`PROJECT_NARRATIVE.md`](./PROJECT_NARRATIVE.md).

System for generating synthetic data with statistical, causal, and logical validation in 8 modules: from the Statistical Contract (Module 1) to the final CSV with Confidence Report (Module 7) and Readiness Report — utility, privacy, and practical value (Module 8).

Open source under [Apache 2.0 license](./LICENSE). Contributions are welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md).


## Install

```bash
git clone https://github.com/synap-project/synap.git
cd synap
pip install -e . # normal installation
pip install -e ".[dev]" # + pytest, ruff (for development)

```

Requires Python 3.11+. This installs the `synap` package and the `synap` command line.

## Run (CLI)

```bash
synap \

--schema path/to/schema.yaml \

--anchor-csv path/to/real_data.csv \

--target-rows 1000000 \

--batch-size 5000 \

--real-holdout-csv path/to/holdout.csv \

--training-reference-csv path/to/training_anchor.csv \

--target-column risk \

--output-dir ./output

```

Module 8 (Auditor) runs automatically after Module 7 releases the output, provided `--real-holdout-csv` is supplied (use
`--no-auditor` to disable it). `--training-reference-csv` is optional
and only necessary for the Membership Inference Attack — without it,
this specific check is marked as "not evaluated" (never
faked).

Without `--anchor-csv`, Module 1 accepts manual parameters (only available via the Python API — see below) or assumes Pure Synthetic Mode (default distributions, with warning).

Without `--real-holdout-csv`, Module 7 blocks output by default
(the Fidelity Index cannot be calculated without real comparison data); use `--force-out` to release it anyway.

## Run (Python API)

```python
from pathlib import Path
from synap.main import SynapConfig, run_synap

config = SynapConfig(
schema_path=Path("schema.yaml"),
anchor_csv=Path("dados_reais.csv"), # ou manual_params={...}
target_rows=1_000_000,
batch_size=5_000,
real_holdout_csv=Path("holdout.csv"),
target_column="risco",
output_dir=Path("./saida"),

rng_seed=42,

resultado = run_synap(config)

if resultado.resultado_compilador.liberado:

print(resultado.resultado_compilador.csv_path)

if resultado.resultado_auditor:
print(resultado.resultado_auditor.aprovado_global, resultado.resultado_auditor.report_path)
else:

print(resultado.resultado_compilador.diagnostico)
```

## Schema Format (`schema.yaml`)

```yaml
columns:
age:
type: numeric
minimum: 18
maximum: 75
yield:
type: numeric
minimum: 0
risk:

type: categorical
categories: [low, medium, high]
rules:

- "yield > 0"
```

## Validation and Reliability Report

`validation/Relatorio_Validacao_Fiabilidade_SYNAP.pdf` — empirical proof, with a known fundamental truth-generating process (not just an arbitrary real sample), that the 8 modules recover the real statistical and causal structure. They respect business rules, are useful, and do not expose real data.

Includes the 170 automated tests, Kolmogorov-Smirnov and
chi-square tests, correlation recovery against fundamental truth, TSTR/TRTR,
and DCR/NNDR/MIA — with the limitations found reported as they were measured.

See `validation/README.md` to reproduce.

## Website — real data generation, in the browser

`site/index.html` is the project's public page: futuristic, animated (particle network, animated synaptic pipeline, counters, scroll reveals), with the Validation Report numbers and comparison with the market — **and a functional "Generate Data" section**, which triggers the real Python pipeline (`synap.main.run_synap`, the 8 complete modules), not a JavaScript reimplementation.

```bash
synap --serve-site
# 🌐 S.Y.N.A.P. — website + real pipeline at http://127.0.0.1:8000/

```

Automatically opens the browser at `http://127.0.0.1:8000/`. From
there, in the "Generate Data" section: define the schema (columns, rules),
optionally load a real anchor/holdout CSV, adjust the
parameters, and press "Generate data" — the browser sends the request to a
small local API (`synap.webserver`), which runs the complete pipeline
in a background thread and returns the Loyalty Index, the privacy verdict, and links to download the `dataset_final.csv` and the **real** PDF reports generated in that execution.

Options: `--site-port 8080` (change port), `--no-open-browser`,
`--repo-dir /alternative/path` (serve another checkout).

The server deliberately serves the repository root (not just `site/`) — the website links to `README.md`,
`PROJECT_NARRATIVE.md`, `CONTRIBUTING.md`,
`validation/Relatorio_Validacao_Fiabilidade_SYNAP.pdf`, etc. are
real relative paths, served locally, without depending on the
project already being published on GitHub.

Alternative, without a functional pipeline (only the static page): publish `site/` via GitHub Pages as soon as the repository exists
(`Settings → Pages → /site`),
or any static hosting.



## Test

``bash
pytest # all modules + integration (170+ tests)
pytest tests/test_integration.py -v # only the end-to-end pipeline
pytest tests/neocortex -v # only Module 1, etc.

pytest --cov=synap --cov-report=html # with coverage report
ruff check src tests # linting
```

## Structure

```
synap/
├── pyproject.toml # packaging (layout src/), entry point `synap`
├── LICENSE # Apache 2.0
├── CONTRIBUTING.md # how to contribute
├── CODE_OF_CONDUCT.md # Contributor Covenant
├── SECURITY.md # responsible disclosure policy
├── CHANGELOG.md # version history
├── CITATION.cff # academic citation
├── .github/
│ ├── workflows/ci.yml # tests + lint in CI, Python 3.11/3.12
│ ├── ISSUE_TEMPLATE/ # bug/functionality templates
│ └── PULL_REQUEST_TEMPLATE.md
├── src/synap/
│ ├── main.py # orchestrator (SynapConfig, run_synap, CLI, --serve-site)
│ ├── webserver.py # Flask API that connects the website to the real pipeline
│ ├── neocortex/ # Module 1 — Statistical Contract
│ ├── hippocampus/ # Module 2 — Causal DAG + Seed
│ ├── expander/ # Module 3 — remixing + Mahalanobis/plausibility
│ ├── hypothalamus/ # Module 4 — ASP + causal drift + ILP
│ ├── nucleus/ # Module 5 — autoencoder + symbolic neural loop
│ │ # (composes M3+M4+M5 in a single cycle per line — see note below)
│ ├── homeostasis/ # Module 6 — KL-divergence + entropy + reweighting
│ ├── compiler/ # Module 7 — final validation + Twin Model + CSV/PDF
│ └── auditor/ # Module 8 — TSTR/TRTR + DCR/NNDR/MIA (privacy has veto)
├── tests/ # one subdirectory per module, plus test_integration.py
├── validation/ # Validation and Reliability Report + scripts to reproduce
└── site/ # Website (presentation + real data generation via API)
```

## Important architecture note

The "two-way cognitive loop" (Modules 3+4+5) is implemented with
row-by-row ASP and consolidated statistical filters in
`synap.nucleo.run_nucleo` (reusing the public parts of Module 3
and the `validate_row` of Module 4), so that rejections model subsequent remixes **within the same batch**. The part of Module 4 responsible for **continuous causal drift and rule induction (ILP)** is distinct from this and runs separately, once per batch/checkpoint, via `synap.hipotalamo.run_hipotalamo`. The rules induced by ILP are not automatically re-injected into ASP (incompatible rule forms — see `SynapResult.warnings` for the candidate rules of each execution).

## Community

- **Contribute**: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- **Code of Conduct**: [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)
- **Security / Privacy**: [`SECURITY.md`](./SECURITY.md) — do not open public issues for vulnerabilities

- **Version History**: [`CHANGELOG.md`](./CHANGELOG.md)
- **Academic Citation**: [`CITATION.cff`](./CITATION.cff)
- **License**: [Apache 2.0](./LICENSE)

> Note: the URLs `github.com/synap-project/synap` in this README are a placeholder — replace with the actual address from the repository as soon as
> the project is published.

