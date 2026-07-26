"""
Per-instance AUROC of the Extremes / Intermediate RQ1 estimators, pooled across
held-out OOD benchmarks. Disambiguates slice-level AEE from per-example
discrimination for the limitations-section robustness check.

Loads each ``src/data/models/{llm}_{group}.joblib`` (+ scaler), scores all
instances from the 8 OOD benchmarks (10 total minus the 2 used for supervision),
and reports pooled AUROC against the ``success`` labels.
"""

import os

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

LLMS = [
    "phi3-3b",
    "ministral3-3b",
    "ministral3-8b",
    "qwen3-4b",
    "qwen3-8b",
    "gemma3-4b",
    "gemma3-12b",
    "llama3-8b",
    "oss-20b",
]

PRETTY = {
    "phi3-3b": "Phi-3.5-Mini (3.6B)",
    "ministral3-3b": "Ministral3 (3B)",
    "ministral3-8b": "Ministral3 (8B)",
    "qwen3-4b": "Qwen3 (4B)",
    "qwen3-8b": "Qwen3 (8B)",
    "gemma3-4b": "Gemma3 (4B)",
    "gemma3-12b": "Gemma3 (12B)",
    "llama3-8b": "Llama-3.1 (8B)",
    "oss-20b": "GPT-OSS (20B)",
}

ALL_BENCHES = [
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

GROUPS = {
    "extremes": ["gsm-test", "olympiadbench-test"],
    "intermediate": ["mathhendrycks-test", "scibench-test"],
}

FEATURE_INDICES = list(range(0, 10))  # mean..kurtosis (10 entropy-distribution stats)


def _load_bench(llm: str, bench: str):
    path = f"src/data/features/{llm}-{bench}.pt"
    if not os.path.exists(path):
        return None, None
    data = torch.load(path)
    X = np.stack([item["features:"][FEATURE_INDICES].numpy() for item in data]).astype(np.float32)
    y = np.array([int(bool(item["success"])) for item in data])
    return X, y


def _safe_auroc(y, p):
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def score_one(llm: str, group: str):
    held_out = [b for b in ALL_BENCHES if b not in GROUPS[group]]
    model_path = f"src/data/models/{llm}_{group}.joblib"
    scaler_path = f"src/data/models/{llm}_{group}_scaler.joblib"
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    per_bench = {}
    X_all, y_all = [], []
    for bench in held_out:
        X, y = _load_bench(llm, bench)
        if X is None:
            continue
        X_s = scaler.transform(X)
        p = model.predict_proba(X_s)[:, 1]
        per_bench[bench] = {
            "auroc": _safe_auroc(y, p),
            "n": int(len(y)),
            "acc": float(np.mean(y)),
        }
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        return None

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    X_s = scaler.transform(X_all)
    p = model.predict_proba(X_s)[:, 1]
    pooled = {
        "auroc": _safe_auroc(y_all, p),
        "n": int(len(y_all)),
        "acc": float(np.mean(y_all)),
    }
    return {"pooled": pooled, "per_bench": per_bench}


def _print_group_grid(rows, group):
    """One table per group: rows = LLMs, columns = held-out benchmarks."""
    held_out = [b for b in ALL_BENCHES if b not in GROUPS[group]]
    short = lambda b: b.replace("-test", "").replace("mathhendrycks", "math").replace("olympiadbench", "olymp").replace("gsmsymbolic", "gsmsym").replace("livemathbench", "lmb").replace("matscibench", "matsci").replace("theoremqa", "thmqa")
    print(f"\n=== Group: {group.upper()} ===")
    header = f"{'Model':<22}"
    for b in held_out:
        header += f"{short(b):>9}"
    header += f"{'pooled':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        r = row.get(group)
        if r is None:
            continue
        line = f"{PRETTY[row['llm']]:<22}"
        for b in held_out:
            pb = r["per_bench"].get(b)
            line += f"{pb['auroc']:>9.3f}" if pb and not np.isnan(pb['auroc']) else f"{'—':>9}"
        line += f"{r['pooled']['auroc']:>9.3f}"
        print(line)

    # Per-benchmark median across LLMs
    print("-" * len(header))
    med_line = f"{'Median across LLMs':<22}"
    for b in held_out:
        vals = [row[group]["per_bench"][b]["auroc"] for row in rows
                if row.get(group) and b in row[group]["per_bench"]
                and not np.isnan(row[group]["per_bench"][b]["auroc"])]
        med_line += f"{np.median(vals):>9.3f}" if vals else f"{'—':>9}"
    pooled_vals = [row[group]["pooled"]["auroc"] for row in rows if row.get(group)]
    med_line += f"{np.median(pooled_vals):>9.3f}" if pooled_vals else f"{'—':>9}"
    print(med_line)


def main():
    rows = []
    for llm in LLMS:
        row = {"llm": llm}
        for group in GROUPS:
            row[group] = score_one(llm, group)
        rows.append(row)

    for group in GROUPS:
        _print_group_grid(rows, group)

    # CSV: long format, one row per (llm, group, benchmark) + pooled rows
    out = "src/analysis/sensibility_predictive_power/per_instance_auroc.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("llm,group,benchmark,auroc,n,acc\n")
        for row in rows:
            for g in GROUPS:
                r = row.get(g)
                if r is None:
                    continue
                for b, pb in r["per_bench"].items():
                    f.write(f"{row['llm']},{g},{b},{pb['auroc']:.6f},{pb['n']},{pb['acc']:.6f}\n")
                f.write(f"{row['llm']},{g},__pooled__,{r['pooled']['auroc']:.6f},{r['pooled']['n']},{r['pooled']['acc']:.6f}\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
