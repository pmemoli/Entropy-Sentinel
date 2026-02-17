# Anonymous Submission for ACL 2026

This repository contains the code for reproducing the experiments in our ACL submission.

## Overview

This project investigates whether entropy-based signatures from language models can effectively estimate accuracy on mathematical and scientific reasoning benchmarks. We evaluate multiple language models across various reasoning tasks and train classifiers to predict performance from internal model signals.

## Project Structure

```
.
├── src/
│   ├── engine/          # Core experiment modules
│   ├── scripts/         # Execution scripts
│   └── data/            # Generated data (excluded from repo)
│       ├── features/    # Extracted entropy feature vectors  
│       ├── models/      # Saved classifier models 
│       └── runs/        # Stored activation profiles
└── requirements.txt     # Python dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (if needed for evaluation):
```bash
cp .env.example .env  # Edit as needed
```

## Reproducing Results

To reproduce the experiments, run the following scripts in order:

### 1. Store Activations
Runs language models on benchmarks and stores activation profiles:
```bash
bash src/scripts/store_activations.sh
```
This processes multiple model-benchmark combinations and saves entropy profiles to `src/data/runs/`.

### 2. Evaluate Runs
Evaluates model performance on the benchmarks:
```bash
bash src/scripts/evaluate_runs.sh
```
Computes accuracy metrics for each model-benchmark pair.

### 3. Generate Features
Extracts statistical features from entropy profiles:
```bash
bash src/scripts/generate_features.sh
```
Generates feature vectors from the stored activations and saves them to `src/data/features/`.

### 4. Train Classifiers
Trains accuracy prediction models:
```bash
python src/scripts/train_classifiers.py
```
Trains multiple classifier configurations (Random Forest, Logistic Regression, Neural Networks) to predict accuracy from entropy features.


### Training Experiments

#### Sensible Defaults
Train classifiers with recommended default settings:
```bash
python src/scripts/train_sensible_defaults.py
```

This script trains classifiers using standard configurations across all models and benchmarks. It provides a baseline for comparison and is the recommended starting point for experiments.

#### Feature Ablation Study
Experiment with different feature subsets:
```bash
python src/scripts/train_different_features.py
```

This script trains classifiers with various feature combinations to identify which entropy-based features are most predictive:
- Single features (se_sum only, max only)
- Baseline combinations (top2_baselines)
- Reduced feature sets

Both scripts automatically skip already-trained models and provide progress tracking.

## Models and Benchmarks

The experiments evaluate the following models:
- Phi-3 (3.8B parameters)
- Qwen3 (4B and 8B parameters)
- Ministral-3 (3B and 8B parameters)
- Llama 3.1 (8B parameters)
- Gemma 3 (4B and 12B parameters)
- GPT-OSS (20B parameters)

On the following benchmarks:
- **Mathematical Reasoning**: GSM8K, MATH (Hendrycks), SVAMP, GSM-Symbolic, LiveMathBench
- **Scientific Reasoning**: GPQA, SciBench, TheoremQA, OlympiadBench, MatSciBench

## Notes

- The scripts process multiple configurations and may take several hours to complete
- Intermediate results are saved to allow resuming if interrupted
- Training scripts automatically skip existing models to enable easy resumption

# List all available features
python -m src.engine.train_calibrator --list_features

# Train on SEA (entropy sum) only
python -m src.engine.train_calibrator \
    --train_suites phi3-3b-gsm-test phi3-3b-olympiadbench-test \
    --model_name phi3_sea_only \
    --feature_subset SE_sum

# Train on all logprob-based metrics
python -m src.engine.train_calibrator \
    --train_suites phi3-3b-gsm-test phi3-3b-olympiadbench-test \
    --model_name phi3_logprob_features \
    --feature_subset nll_avg nll_max nll_sum lntp mtp ppl

# Train on all features
python -m src.engine.train_calibrator \
    --train_suites phi3-3b-gsm-test phi3-3b-olympiadbench-test \
    --model_name phi3_all_features

# Use legacy feature names (backwards compatible)
python -m src.engine.train_calibrator \
    --train_suites phi3-3b-gsm-test \
    --model_name phi3_legacy \
    --feature_subset se_sum max mean
