# Entropy Sentinel

This repository contains the code for reproducing the experiments from the paper ["Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM"](https://arxiv.org/abs/2601.09001).

## Overview

Entropy Sentinel investigates whether the **decoding entropy trace** of a language model — the sequence of Shannon entropies of the token distribution at each generation step — carries enough signal to estimate whether an answer is correct, without access to a ground-truth label at inference time.

For every generation we store the entropy profile and the selected-token log-probabilities, summarize them into a compact feature vector (distribution statistics plus standard uncertainty-quantification baselines), and train lightweight classifiers/regressors to predict a judged quality label. The project covers two settings:

- **STEM** — binary correctness (`success`) on mathematical and scientific reasoning benchmarks, labeled by an LLM judge.
- **Monitoring** — a 1–10 quality score on open-ended multi-turn conversations (WildBench), labeled by an LLM judge, evaluated for out-of-distribution generalization across topic categories (leave-one-category-out).

## Data

The generated data (model runs, extracted features, trained models, sensitivity sweeps) is **not** stored in this repository — it is large and lives on Google Drive:

**https://drive.google.com/drive/u/0/folders/1GiVLBkOZktWzH863FVqBIs_kBoDQeGDA**

Download it and place it at `src/data/` so the tree looks like:

```
src/data/
├── runs/          # Raw generations + entropy profiles per suite (.pt), from run-* steps
├── features/      # Extracted feature vectors per suite (<suite>.pt), from features-* steps
├── models/        # Trained classifier/regressor artifacts (.joblib), from classifier steps
├── sensitivity/   # Temperature/seed sweep runs (STEM), from sensitivity-stem
└── audit.csv      # Manual human audit of LLM-judge labels
```

`src/data/` is git-ignored. If you regenerate everything from scratch (see below) these directories are created automatically.

## Project Structure

```
.
├── Makefile                 # Entry points for every pipeline stage (see below)
├── pyproject.toml           # Dependencies (managed with uv)
├── src/
│   ├── engine/              # Core experiment logic
│   │   ├── core/            # Entropy/feature computations, prompts, types
│   │   ├── scenarios/       # Benchmark loaders (registry of datasets)
│   │   ├── run_stem_scenarios.py         # Generate + store entropy traces (STEM)
│   │   ├── run_monitoring_scenarios.py   # Generate + store entropy traces (monitoring)
│   │   ├── judge_stem_scenarios.py       # LLM-judge correctness labeling (STEM)
│   │   ├── judge_monitoring_scenarios.py # LLM-judge 1–10 scoring (monitoring)
│   │   ├── generate_stem_features.py     # Feature extraction (STEM)
│   │   ├── generate_monitoring_features.py
│   │   ├── train_stem_entropy_sentinel.py
│   │   └── train_monitoring_entropy_sentinel.py
│   ├── scripts/             # Batch drivers over the full suite list (invoked by the Makefile)
│   ├── analysis/            # Figure/table generation from results
│   ├── results/             # Committed figures, tables, and summary CSVs
│   └── data/                # Generated data — download from Google Drive (git-ignored)
└── tests/                   # Scenario/loader tests
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.10.

1. Install dependencies:
```bash
uv sync
```

2. Create a `.env` file in the repository root with the credentials the pipeline needs:
```bash
HF_TOKEN=...           # Hugging Face token (required for gated models: Llama, Gemma, Ministral)
XAI_API_KEY=...        # xAI key — LLM judge for STEM correctness
OPENAI_API_KEY=...     # OpenAI key — LLM judge for monitoring scores
# Optional, depending on which judges/models you enable:
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
HOST=...               # Only used by the `make sync`/`make ssh` remote-sync helpers
```

Generation runs on GPU via vLLM, so a CUDA-capable machine is required to reproduce the runs. `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` can be set in the environment or passed through to the Makefile.

## Reproducing Results

If you only want to analyze existing results, download `src/data/` from Google Drive and skip to [Analysis](#analysis). To regenerate from scratch, run the pipeline stages below. Each stage iterates over the full model × benchmark suite list defined in the corresponding driver script.

### STEM pipeline

```bash
make run-stem          # Run models on STEM benchmarks; store entropy traces to src/data/runs/
make judge-stem        # Label each generation correct/incorrect with the LLM judge
make features-stem     # Extract feature vectors to src/data/features/
make classifiers-stem  # Train Random Forest / Logistic Regression / Neural Network predictors
```

`make pipeline-stem` runs `run-stem`, `judge-stem`, and `features-stem` in sequence.

Additional STEM targets:

```bash
make audit-stem        # Interactive human audit of judge labels (writes src/data/audit.csv)
make sensitivity-stem  # Temperature × seed sweep on MATH (Phi-3.5-mini) → src/data/sensitivity/
```

### Monitoring pipeline

```bash
make run-monitoring       # Run models on WildBench (multi-turn); store entropy traces
make judge-monitoring     # Score each response 1–10 with the LLM judge
make features-monitoring  # Extract feature vectors
```

`make pipeline-monitoring` runs the three monitoring stages in sequence. Monitoring classifiers are trained with:

```bash
uv run python -m src.scripts.train_monitoring_classifiers
```

### Analysis

The scripts in `src/analysis/` regenerate the figures, tables, and summary CSVs committed under `src/results/`:

```bash
uv run python -m src.analysis.stemqa_analysis            # STEM figures/tables
uv run python -m src.analysis.monitoring_analysis        # Monitoring (LOCO) figures/tables
uv run python -m src.analysis.temp_sensitivity_analysis  # Temperature/seed sensitivity
```

## Models and Benchmarks

**Models evaluated:**

- Phi-3.5-mini-instruct (~3.8B)
- Qwen3 (4B and 8B)
- Ministral-3 (3B and 8B)
- Llama-3.1 (8B)
- Gemma-3 (4B and 12B)
- GPT-OSS (20B)

**STEM benchmarks:** GSM8K, MATH (Hendrycks), SVAMP, GSM-Symbolic, LiveMathBench, GPQA, SciBench, TheoremQA, OlympiadBench, MatSciBench.

**Monitoring benchmark:** WildBench (with MT-Bench and Arena-Hard loaders also available in the registry).

## Features

Each generation is summarized into a 17-dimensional feature vector:

- **Entropy distribution summaries (10):** `mean`, `std`, `max`, `q10`, `q25`, `q50`, `q75`, `q90`, `skewness`, `kurtosis`.
- **Uncertainty-quantification baselines (7):** `se_sum`, `nll_avg`, `nll_max`, `nll_sum`, `lntp`, `mtp`, `ppl`.

Ablations train on subsets of these (e.g. entropy-only vs. entropy + UQ baselines).

## Notes

- The generation stages process many model × benchmark combinations and can take several hours on GPU.
- Judge and feature stages skip suites that fail, and training skips artifacts that already exist, so interrupted runs can be resumed.
- `make sync` / `make first-sync` / `make terminate-sync` / `make ssh` are convenience helpers for mirroring the working tree to a remote GPU host via [mutagen](https://mutagen.io/) (they require `HOST` in `.env`) and are not needed to reproduce results.
```