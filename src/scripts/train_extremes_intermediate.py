"""
Train only the two paper-named training groups, one model per (LLM, group).

Per Section 5/RQ1 of the paper:
  - Extremes      = GSM8K + OlympiadBench       (widest difficulty span)
  - Intermediate  = MATH (Hendrycks) + SciBench (mid-range difficulty)

Default RQ1 estimator config is used verbatim:
  - random_forest
  - class balancing on
  - calibration on
  - 10 entropy-distribution statistics as the feature subset
    (mean, std, max, q10..q90, skewness, kurtosis)

Outputs to ``src/data/models/{llm}_{group}.joblib`` (+ scaler), so analysis
scripts can load each (LLM, group) model by name without computing a hash.
"""

import os
import torch

from src.engine.train_classifier import run_training

LLMS = [
    "phi3-3b",
    "qwen3-4b",
    "qwen3-8b",
    "ministral3-3b",
    "ministral3-8b",
    "gemma3-4b",
    "gemma3-12b",
    "llama3-8b",
    "oss-20b",
]

GROUPS = {
    "extremes": ["gsm-test", "olympiadbench-test"],
    "intermediate": ["mathhendrycks-test", "scibench-test"],
}

FEATURE_SUBSET = [
    "mean", "std", "max",
    "q10", "q25", "q50", "q75", "q90",
    "skewness", "kurtosis",
]

MODELS_DIR = "src/data/models"
FEATURES_DIR = "src/data/features"


def load_suite_cache():
    cache = {}
    needed = {
        f"{llm}-{bench}"
        for llm in LLMS
        for benches in GROUPS.values()
        for bench in benches
    }
    for key in sorted(needed):
        path = f"{FEATURES_DIR}/{key}.pt"
        if os.path.exists(path):
            cache[key] = torch.load(path)
        else:
            print(f"  WARN: missing features file {path}")
    return cache


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Loading feature files...")
    suite_cache = load_suite_cache()
    print(f"Loaded {len(suite_cache)} feature files.")

    total = len(LLMS) * len(GROUPS)
    i = 0
    for llm in LLMS:
        for group_name, benches in GROUPS.items():
            i += 1
            model_name = f"{llm}_{group_name}"
            train_suites = [f"{llm}-{b}" for b in benches]
            missing = [s for s in train_suites if s not in suite_cache]
            if missing:
                print(
                    f"[{i}/{total}] Skipping {model_name}: missing features "
                    f"{missing}"
                )
                continue

            out_path = f"{MODELS_DIR}/{model_name}.joblib"
            if os.path.exists(out_path):
                print(f"[{i}/{total}] Skipping {model_name} (already trained)")
                continue

            print(f"[{i}/{total}] Training {model_name} on {train_suites}")
            run_training(
                train_suites=train_suites,
                model="random_forest",
                model_name=model_name,
                balance_classes=True,
                calibrate=True,
                feature_subset=FEATURE_SUBSET,
                suite_cache=suite_cache,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
