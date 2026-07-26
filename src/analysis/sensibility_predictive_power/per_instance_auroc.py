"""
Per-instance AUROC of the Extremes / Intermediate RQ1 estimators, per benchmark.

For each (LLM, group, benchmark):
  - If the benchmark is OOD (not in the training group), score the full benchmark.
  - If the benchmark is in the training group, score the seed-42 held-out 20%
    split — the in-domain portion the trained model never saw. (Same split
    rule as train_classifier.py:62-67.)

Outputs:
  - Console: benchmark-rows median + [Q25, Q75] across LLMs per group.
  - CSV: src/analysis/sensibility_predictive_power/per_instance_auroc.csv
         long format, one row per (llm, group, benchmark).
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

PRETTY_BENCH = {
    "gsm-test": "GSM",
    "gsmsymbolic-test": "GSM-Symbolic",
    "svamp-test": "SVAMP",
    "mathhendrycks-test": "MATH",
    "olympiadbench-test": "OlympiadBench",
    "gpqa-test": "GPQA",
    "scibench-test": "SciBench",
    "theoremqa-test": "TheoremQA",
    "livemathbench-test": "LiveMathBench",
    "matscibench-test": "MatSciBench",
}

ALL_BENCHES = list(PRETTY_BENCH.keys())

GROUPS = {
    "extremes": ["gsm-test", "olympiadbench-test"],
    "intermediate": ["mathhendrycks-test", "scibench-test"],
}

FEATURE_INDICES = list(range(0, 10))  # 10 entropy-distribution stats


def _load_bench(llm: str, bench: str, slice_: str):
    """slice_: "all" or "test_20" (seed-42 held-out 20%)."""
    path = f"src/data/features/{llm}-{bench}.pt"
    if not os.path.exists(path):
        return None, None
    data = torch.load(path)
    if slice_ == "test_20":
        # Mirror train_classifier.py:62-67 exactly so we get the leftover 20%.
        np.random.seed(42)
        np.random.shuffle(data)
        split_index = int(0.8 * len(data))
        data = data[split_index:]
    if not data:
        return None, None
    X = np.stack([item["features:"][FEATURE_INDICES].numpy() for item in data]).astype(np.float32)
    y = np.array([int(bool(item["success"])) for item in data])
    return X, y


def _score(model, scaler, X, y):
    p = model.predict_proba(scaler.transform(X))[:, 1]
    auroc = float("nan") if len(set(y.tolist())) < 2 else float(roc_auc_score(y, p))
    aee = float(abs(float(np.mean(p)) - float(np.mean(y))))
    return auroc, aee


def score_all():
    """Returns dict: (llm, group, bench) -> {auroc, n, in_domain}."""
    out = {}
    for llm in LLMS:
        for group, train_benches in GROUPS.items():
            model_path = f"src/data/models/{llm}_{group}.joblib"
            scaler_path = f"src/data/models/{llm}_{group}_scaler.joblib"
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            for bench in ALL_BENCHES:
                in_domain = bench in train_benches
                slice_ = "test_20" if in_domain else "all"
                X, y = _load_bench(llm, bench, slice_)
                if X is None:
                    continue
                auroc, aee = _score(model, scaler, X, y)
                out[(llm, group, bench)] = {
                    "auroc": auroc,
                    "aee": aee,
                    "n": int(len(y)),
                    "in_domain": in_domain,
                }
    return out


def _stats(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return None
    return {
        "median": float(np.median(vals)),
        "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
        "n_llms": len(vals),
    }


def summarize_by_bench(scores):
    """For each (group, bench): median + IQR of AUROC and AEE across LLMs."""
    summary = {}
    for group in GROUPS:
        for bench in ALL_BENCHES:
            keys = [llm for llm in LLMS if (llm, group, bench) in scores]
            if not keys:
                continue
            in_domain = bench in GROUPS[group]
            summary[(group, bench)] = {
                "auroc": _stats([scores[(llm, group, bench)]["auroc"] for llm in keys]),
                "aee": _stats([scores[(llm, group, bench)]["aee"] for llm in keys]),
                "in_domain": in_domain,
            }
    return summary


def _fmt(stat):
    if stat is None:
        return "—"
    return f"{stat['median']:.3f}_{{{stat['iqr']:.3f}}}"


def print_table(summary):
    print(
        f"\n{'Benchmark':<18} "
        f"{'AEE-Ext':>14} {'AUROC-Ext':>14} "
        f"{'AEE-Int':>14} {'AUROC-Int':>14}"
    )
    print("-" * 78)
    for bench in ALL_BENCHES:
        line = f"{PRETTY_BENCH[bench]:<18} "
        for group in ("extremes", "intermediate"):
            s = summary.get((group, bench))
            if s is None:
                line += f"{'—':>14} {'—':>14} "
                continue
            tag = "*" if s["in_domain"] else " "
            line += f"{_fmt(s['aee']):>13}{tag} {_fmt(s['auroc']):>13}{tag} "
        print(line)
    print("* = seed-42 held-out 20% of an in-domain (training) benchmark")


def save_csv(scores, summary):
    long_csv = "src/analysis/sensibility_predictive_power/per_instance_auroc.csv"
    os.makedirs(os.path.dirname(long_csv), exist_ok=True)
    with open(long_csv, "w") as f:
        f.write("llm,group,benchmark,auroc,aee,n,in_domain\n")
        for (llm, group, bench), r in scores.items():
            f.write(
                f"{llm},{group},{bench},{r['auroc']:.6f},{r['aee']:.6f},"
                f"{r['n']},{int(r['in_domain'])}\n"
            )
    print(f"\nSaved {long_csv}")

    summary_csv = "src/analysis/sensibility_predictive_power/per_instance_auroc_summary.csv"
    with open(summary_csv, "w") as f:
        f.write(
            "group,benchmark,"
            "auroc_median,auroc_iqr,auroc_q25,auroc_q75,"
            "aee_median,aee_iqr,aee_q25,aee_q75,"
            "n_llms,in_domain\n"
        )
        for (group, bench), s in summary.items():
            a, e = s["auroc"], s["aee"]
            f.write(
                f"{group},{bench},"
                f"{a['median']:.6f},{a['iqr']:.6f},{a['q25']:.6f},{a['q75']:.6f},"
                f"{e['median']:.6f},{e['iqr']:.6f},{e['q25']:.6f},{e['q75']:.6f},"
                f"{a['n_llms']},{int(s['in_domain'])}\n"
            )
    print(f"Saved {summary_csv}")


def main():
    scores = score_all()
    summary = summarize_by_bench(scores)
    print_table(summary)
    save_csv(scores, summary)


if __name__ == "__main__":
    main()
