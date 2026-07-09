# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/pt-PT/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [0.1.0] — 2026-07-09

### Added

Initial open-source release. Complete synthetic data generation system, in 8 modules:

- **Module 1 — Neocortex Perceptual**: Statistical Contract from real data (CSV) or manual parameters; insufficient sample detection; Pure Synthetic Mode with explicit warning.

- **Module 2 — Hippocampus**: validation/adjustment of causal DAG by d-separation test; DAG machine learning (simplified PC-algorithm); initial seed generation by conditional linear-Gaussian sampling.

- **Module 3 — Expander**: compositional remixing with vector memory; Mahalanobis distance filter; plausibility score (real-vs-scrambled random forest); dynamic negative attention.

- **Module 4 — Hypothalamus**: vectorized strict logical validation (ASP); continuous causal drift monitoring; rule induction (ILP) with logical consistency validation.

- **Module 5 — Cognitive Core**: artifact detection by autoencoder (loss of reconstruction); neural-symbolic loop composing Mahalanobis + plausibility + ASP + line-by-line autoencoder.
- **Module 6 — Homeostasis**: Kullback-Leibler divergence and Shannon entropy monitored by EMA; reweighting of sources by likelihood ratio; recommendation to pause at critical drift.

- **Module 7 — Compiler**: exhaustive final ASP validation; Twin Model (Logistic/Linear Regression); Fidelity Index; Confidence Report in PDF.

- **Module 8 — Auditor**: multi-task TSTR/TRTR utility; privacy via Distance to Closest Record, Nearest-Neighbour Distance Ratio, exact matching, and Membership Inference Attack — privacy with veto power over statistical fidelity.

- **Orchestrator** (`synap.main`): `SynapConfig`/`run_synap`, CLI
(`synap --schema ... --target-rows ...`; `synap --serve-site` serves
the website AND the actual pipeline API).

- **Web Server** (`synap.webserver`): Flask API (`/api/generate`,
`/api/jobs/<id>`, `/api/jobs/<id>/download/<file>`) that connects the
website to the actual Python pipeline — data generation from the
browser triggers the 8 modules in fact, not a reimplementation in
JavaScript. Jobs run in background threads; the website polls
the state and displays the Loyalty Index, the privacy verdict, and actual download links.

- **Website** (`site/index.html`): futuristic, animated (network of particles, synaptic pipeline, counters, scroll reveals), with the Validation Report numbers — and a functional "Generate Data" section, connected to the actual pipeline via `synap.webserver`.

- **170 automated tests** (pytest) covering the 8 modules and end-to-end integration.

- **Validation and Reliability Report** (`validation/`): empirical proof with a known fundamental truth generator process — Kolmogorov-Smirnov, chi-square, correlation recovery, TSTR/TRTR and DCR/NNDR/MIA tests.

- Packaging as an installable Python project (`pyproject.toml`, layout `src/`), with the `synap` entry point on the command line.

### Known (documented) limitations

- Simplified causal learning (PC-algorithm) — does not fully implement Meek's rules.

- Rule induction engine (ILP) restricted to a univariate template.

- Categorical columns only link to numeric columns through an explicitly provided DAG
— without a DAG, these dependencies do not survive (quantified in the Validation Report, Section 4).
