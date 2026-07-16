import hashlib
import json
import os
import torch
from itertools import product, combinations
from src.engine.train_stem_entropy_sentinel import run_training, Model

train_suites = [
    "gsm-test",
    "gsmsymbolic-test",
    "svamp-test",
    "mathhendrycks-test",
    "olympiadbench-test",
    "gpqa-test",
    "scibench-test",
    "theoremqa-test",
    "livemathbench-test",
    "matscibench-test",
]

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

classifiers: list[Model] = [
    "random_forest",
    "logistic_regression",
    "neural_network",
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

feature_families = [
    list(range(0, len(feature_names))),  # All features
    list(range(0, 10)),  # Statistics features
    [2, 10, 13],  # max, se sum and nll sum
    [10],  # se sum only
]

benchmark_groups = []
for size in range(1, 5):
    for combo in combinations(train_suites, size):
        benchmark_groups.append(list(combo))

balance_classes = [True, False]
calibrate = [True, False]

total_models = (
    len(llms)
    * len(benchmark_groups)
    * len(classifiers)
    * len(calibrate)
    * len(balance_classes)
    * len(feature_families)
)

# Pre-load all feature files into memory to avoid repeated disk reads
print("Loading feature files into memory...")
suite_cache = {}
for llm in llms:
    for benchmark in train_suites:
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

for llm, benchmarks, clf, cal, bal, feat_idx in product(
    llms,
    benchmark_groups,
    classifiers,
    calibrate,
    balance_classes,
    feature_families,
):
    current += 1

    benchmarks = sorted(benchmarks)
    feat_idx = sorted(feat_idx)

    config = {
        "llm": llm,
        "benchmarks": benchmarks,
        "train_suites": [f"{llm}-{s}" for s in benchmarks],
        "classifier": clf,
        "calibrate": cal,
        "balance_classes": bal,
        "features": [feature_names[i] for i in feat_idx],
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
        model=clf,
        model_name=hash_val,
        balance_classes=bal,
        calibrate=cal,
        feature_subset=config["features"],
        suite_cache=suite_cache,
    )

    completed += 1

print(f"\nCompleted: {completed}")
print(f"Skipped: {skipped}")
print(f"Total configs: {total_models}")
