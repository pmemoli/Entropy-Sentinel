import hashlib
import json
import os
from itertools import product
from src.engine.train_baseline import run_training

llms = [
    "qwen3-8b",
    "phi3-3b",
    "qwen3-4b",
    "ministral3-3b",
    "ministral3-8b",
    "llama3-8b",
    "gemma3-4b",
    "gemma3-12b",
    "oss-20b",
]

feature_names = [
    "mean",
    "std",
    "max",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "skewness",
    "kurtosis",
    "se_sum",
    "nll_avg",
    "nll_max",
    "nll_sum",
    "lntp",
    "mtp",
    "ppl",
]

benchmark_groups = [["gsm-test", "olympiadbench-test"]]
total_models = len(llms) * len(benchmark_groups) * len(feature_names)

# Read hashes from model directory
files = os.listdir("src/data/models/")
existing_hashes = set()
for file in files:
    if file.endswith(".joblib") and "scaler" not in file:
        existing_hashes.add(file.replace(".joblib", ""))

print(f"Found {len(existing_hashes)} existing models.")

skipped = 0
completed = 0
current = 0

for llm, benchmarks, feat in product(
    llms,
    benchmark_groups,
    feature_names,
):
    current += 1

    benchmarks = sorted(benchmarks)

    config = {
        "llm": llm,
        "benchmarks": benchmarks,
        "train_suites": [f"{llm}-{s}" for s in benchmarks],
        "classifier": "raw_logistic_regression",
        "calibrate": False,
        "balance_classes": False,
        "features": [feat],
    }

    hash_val = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]

    if hash_val in existing_hashes:
        print(
            f"[{current}/{total_models}] Skipping {hash_val} (already trained)"
        )
        skipped += 1
        continue

    print(f"[{current}/{total_models}] Training {hash_val}")

    run_training(
        train_suites=config["train_suites"],
        model_name=hash_val,
        feature=feat,
    )

    completed += 1

print(f"\nCompleted: {completed}")
print(f"Skipped: {skipped}")
print(f"Total configs: {total_models}")
