import hashlib
import json
import os
import torch
from src.engine.train_monitoring_entropy_sentinel import run_training, Model

benchmarks = [
    "wildbench-test",
]

llms = [
    "phi3-3b",
]

classifier: Model = "random_forest"

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

# The entropy distribution summaries, without the UQ baseline metrics.
features = feature_names[:10]

total_models = len(llms)

# Pre-load all feature files into memory to avoid repeated disk reads
print("Loading feature files into memory...")
suite_cache = {}
for llm in llms:
    for benchmark in benchmarks:
        suite_key = f"{llm}-{benchmark}"
        path = f"src/data/features/{suite_key}.pt"
        if os.path.exists(path):
            suite_cache[suite_key] = torch.load(path)
print(f"Loaded {len(suite_cache)} feature files.")

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

for llm in llms:
    current += 1

    config = {
        "llm": llm,
        "benchmarks": sorted(benchmarks),
        "train_suites": [f"{llm}-{s}" for s in sorted(benchmarks)],
        "classifier": classifier,
        "features": features,
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

    print(f"[{current}/{total_models}] Running {hash_val}.")

    run_training(
        train_suites=config["train_suites"],
        model=classifier,
        model_name=hash_val,
        feature_subset=config["features"],
        suite_cache=suite_cache,
    )

    completed += 1

print(f"\nCompleted: {completed}")
print(f"Skipped: {skipped}")
print(f"Total configs: {total_models}")
