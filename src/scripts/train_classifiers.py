import hashlib
import json
import os
from itertools import product, combinations
from src.engine.train_classifier import run_training, Model

train_suites = [
    "gsm-test",
    "gsmsymbolic-test",
    "svamp-test",
    "mathhendrycks-test",
    "olympiadbench-test",
    "gpqa-test",
    "scibench-test",
    "matscibench-test",
    "theoremqa-test",
    "livemathbench-test",
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

benchmark_groups = []
for size in range(9, 10):
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
)

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

for llm, benchmarks, clf, cal, bal in product(
    llms, benchmark_groups, classifiers, calibrate, balance_classes
):
    current += 1

    benchmarks = sorted(benchmarks)

    config = {
        "llm": llm,
        "benchmarks": benchmarks,
        "train_suites": [f"{llm}-{s}" for s in benchmarks],
        "classifier": clf,
        "calibrate": cal,
        "balance_classes": bal,
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

    print(
        f"[{current}/{total_models}] Running {hash_val}: {llm} + {clf} + {len(benchmarks)} benches (cal={cal}, bal={bal})"
    )

    run_training(
        train_suites=config["train_suites"],
        model=clf,
        model_name=hash_val,
        balance_classes=bal,
        calibrate=cal,
    )

    completed += 1

print(f"\nCompleted: {completed}")
print(f"Skipped: {skipped}")
print(f"Total configs: {len(configs)}")
