import csv
import hashlib
import json
import os
import numpy as np
import torch
from itertools import product, combinations
from scipy.stats import spearmanr
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

# Bulk sweep: persisting a .joblib pair per config would be hundreds of GB
# (see diagnosis - calibrated random forest alone stores 5 full 100-tree
# forests per config via CalibratedClassifierCV's internal refitting). The
# CSV of (config, rho, mae) is the actual deliverable; models are treated as
# disposable/regenerable (retraining a single config is ~1-15s).
SAVE_MODELS = False

results_path = "src/results/stem_classifier_evaluation_results.csv"
csv_fields = [
    "hash",
    "llm",
    "benchmarks",
    "classifier",
    "calibrate",
    "balance_classes",
    "features",
    "spearman_rho",
    "mae",
]

total_models = (
    len(llms)
    * len(benchmark_groups)
    * len(classifiers)
    * len(calibrate)
    * len(balance_classes)
    * len(feature_families)
)

# Pre-load all feature files into memory to avoid repeated disk reads.
# NOTE: run_training() copies suite data before shuffling it (rather than
# mutating these cached lists in place), so the same cached list can safely
# be reused - and reshuffled deterministically - across every config that
# trains on that suite.
print("Loading feature files into memory...")
suite_cache = {}
for llm in llms:
    for benchmark in train_suites:
        suite_key = f"{llm}-{benchmark}"
        path = f"src/data/features/{suite_key}.pt"
        if os.path.exists(path):
            suite_cache[suite_key] = torch.load(path)
print(f"Loaded {len(suite_cache)} feature files.")


def evaluate_config(clf, scaler, llm, benchmarks, feature_indices):
    """Domain-level rho/mae across all 10 benchmarks: benchmarks used for
    training are evaluated on their held-out 20% test split (seed 42, same
    split run_training used for the 80% it trained on); benchmarks not used
    for training are evaluated on the full suite (fully OOD)."""
    real_accs, est_accs = {}, {}

    for suite in train_suites:
        suite_key = f"{llm}-{suite}"
        if suite_key not in suite_cache:
            continue

        suite_data = list(suite_cache[suite_key])

        if suite in benchmarks:
            split_index = int(0.8 * len(suite_data))
            np.random.seed(42)
            np.random.shuffle(suite_data)
            suite_data = suite_data[split_index:]

        if len(suite_data) == 0:
            continue

        X_raw = (
            torch.stack(
                [item["features:"][feature_indices] for item in suite_data]
            )
            .float()
            .numpy()
        )
        y = np.array([item["success"] for item in suite_data])

        probs = clf.predict_proba(scaler.transform(X_raw))[:, 1]

        real_accs[suite] = float(np.mean(y))
        est_accs[suite] = float(np.mean(probs))

    if len(real_accs) < 2:
        return None

    real_vals = list(real_accs.values())
    est_vals = list(est_accs.values())

    rho = spearmanr(real_vals, est_vals)[0]
    mae = float(np.mean(np.abs(np.array(real_vals) - np.array(est_vals))))

    return float(rho), mae


existing_hashes = set()
if os.path.exists(results_path):
    with open(results_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            existing_hashes.add(row["hash"])
    print(f"Found {len(existing_hashes)} already-evaluated configs in CSV.")

write_header = not os.path.exists(results_path)
csv_file = open(results_path, "a", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
if write_header:
    csv_writer.writeheader()

skipped = 0
completed = 0
failed = 0
current = 0

for llm, benchmarks, clf_name, cal, bal, feat_idx in product(
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
    feat_subset = [feature_names[i] for i in feat_idx]

    config = {
        "llm": llm,
        "benchmarks": benchmarks,
        "train_suites": [f"{llm}-{s}" for s in benchmarks],
        "classifier": clf_name,
        "calibrate": cal,
        "balance_classes": bal,
        "features": feat_subset,
    }

    hash_val = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]

    if hash_val in existing_hashes:
        print(f"[{current}/{total_models}] Skipping {hash_val} (already evaluated)")
        skipped += 1
        continue

    print(f"[{current}/{total_models}] Running {hash_val}.")

    try:
        trained_model, scaler = run_training(
            train_suites=config["train_suites"],
            model=clf_name,
            model_name=hash_val,
            balance_classes=bal,
            calibrate=cal,
            feature_subset=feat_subset,
            suite_cache=suite_cache,
            save_model=SAVE_MODELS,
        )

        if trained_model is None:
            print(f"  Skipped: not enough samples for cross-validation.")
            failed += 1
            continue

        result = evaluate_config(
            trained_model, scaler, llm, benchmarks, feat_idx
        )

        if result is None:
            print(f"  Skipped: fewer than 2 evaluable benchmarks.")
            failed += 1
            continue

        rho, mae = result

    except Exception as e:
        print(f"  Error evaluating {hash_val}: {e}")
        failed += 1
        continue

    csv_writer.writerow(
        {
            "hash": hash_val,
            "llm": llm,
            "benchmarks": " ".join(benchmarks),
            "classifier": clf_name,
            "calibrate": cal,
            "balance_classes": bal,
            "features": " ".join(feat_subset),
            "spearman_rho": rho,
            "mae": mae,
        }
    )
    csv_file.flush()

    completed += 1

csv_file.close()

print(f"\nCompleted: {completed}")
print(f"Skipped (already evaluated): {skipped}")
print(f"Failed: {failed}")
print(f"Total configs: {total_models}")
