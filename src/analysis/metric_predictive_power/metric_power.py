"""
OG metric-predictive-power AUROC table.

Reads saved runs from ``src/data/runs/{llm}-{benchmark}/*.pt`` and reports the
per-(LLM, benchmark) AUROC of each entropy-profile statistic and each strong
baseline (SE_sum, NLL_*, LNTP, MTP, PPL) at distinguishing successful from
failed generations.

No model inference: uses the top-20-unnormalized ``entropy_profile`` and the
top-20 ``selected_logprobs`` that ``store_activations.py`` wrote at generation
time. This reproduces the table reported in ``src/analysis/section3.ipynb``.

Output: ``src/analysis/metric_predictive_power/og_auroc_table.csv``
"""

import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

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

DATASETS = ["gsm-test", "mathhendrycks-test", "olympiadbench-test"]

METRIC_NAMES = [
    "Mean", "STD", "Max",
    "Q10", "Q25", "Q50", "Q75", "Q90",
    "Skew", "Kurt",
    "SE_sum",
    "NLL_avg", "NLL_max", "NLL_sum",
    "LNTP", "MTP", "PPL",
]

# Flip to 1-AUROC: LNTP/MTP because higher prob = more confident; Skew/Kurt to
# match the failure-side AUROC reported in the paper.
_FLIP_AUC = {"LNTP", "MTP", "Skew", "Kurt"}

OUT_DIR = "src/analysis/metric_predictive_power"


def per_sample_metrics(profile: np.ndarray, selected_logprobs: np.ndarray) -> dict:
    """All 17 per-sample features from one generation.

    ``profile`` is the per-step entropy trajectory; ``selected_logprobs`` is
    log P(sampled token) at each step. Both length T.
    """
    nll = -selected_logprobs
    return {
        "Mean":    float(np.mean(profile)),
        "STD":     float(np.std(profile)),
        "Max":     float(np.max(profile)),
        "Q10":     float(np.percentile(profile, 10)),
        "Q25":     float(np.percentile(profile, 25)),
        "Q50":     float(np.percentile(profile, 50)),
        "Q75":     float(np.percentile(profile, 75)),
        "Q90":     float(np.percentile(profile, 90)),
        "Skew":    float(skew(profile)),
        "Kurt":    float(kurtosis(profile)),
        "SE_sum":  float(np.sum(profile)),
        "NLL_avg": float(np.mean(nll)),
        "NLL_max": float(np.max(nll)),
        "NLL_sum": float(np.sum(nll)),
        "LNTP":    float(np.exp(np.mean(selected_logprobs))),
        "MTP":     float(np.exp(np.min(selected_logprobs))),
        "PPL":     float(np.exp(np.mean(nll))),
    }


def aurocs_from_metrics(per_sample: list[dict], labels: list[int]) -> dict:
    """labels[i] = 1 iff generation i FAILED — higher metric ⇒ higher AUROC."""
    out = {}
    for name in METRIC_NAMES:
        scores = np.array([s[name] for s in per_sample], dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(
                scores, nan=float(np.nanmean(scores)), posinf=1e10, neginf=-1e10
            )
        auc = float(roc_auc_score(labels, scores))
        if name in _FLIP_AUC:
            auc = 1.0 - auc
        out[name] = auc
    return out


def iter_samples_from_dir(runs_dir: str) -> Iterable[dict]:
    if not os.path.isdir(runs_dir):
        return
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith(".pt"):
            continue
        try:
            batch = torch.load(
                os.path.join(runs_dir, fname),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as e:
            print(f"  WARN: skipping {fname}: {e}")
            continue
        for item in batch:
            yield item


def auroc_row_for_suite(suite_dir: str):
    per_sample = []
    labels = []
    for item in iter_samples_from_dir(suite_dir):
        profile = item["entropy_profile"].to(torch.float32).numpy()
        lp = item["selected_logprobs"]
        if isinstance(lp, torch.Tensor):
            lp = lp.to(torch.float32).numpy()
        else:
            lp = np.asarray(lp, dtype=np.float32)
        per_sample.append(per_sample_metrics(profile, lp))
        labels.append(0 if item["success"] else 1)
    if not per_sample:
        return None
    return aurocs_from_metrics(per_sample, labels), len(per_sample)


def build_table(llms, datasets, runs_root) -> pd.DataFrame:
    rows = []
    for llm in llms:
        for ds in datasets:
            suite = f"{llm}-{ds}"
            suite_dir = os.path.join(runs_root, suite)
            print(f"Processing {suite}...", end=" ", flush=True)
            res = auroc_row_for_suite(suite_dir)
            if res is None:
                print("no samples, skipped.")
                continue
            aurocs, n = res
            print(f"n={n}")
            rows.append({"LLM": llm, "Benchmark": ds, "n_samples": n, **aurocs})
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="OG metric-predictive-power AUROC table (no inference).",
    )
    p.add_argument("--runs_root", default="src/data/runs")
    p.add_argument("--output", default=os.path.join(OUT_DIR, "og_auroc_table.csv"))
    p.add_argument("--llms", nargs="+", default=LLMS)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = build_table(args.llms, args.datasets, args.runs_root)
    df.to_csv(args.output, index=False)
    print(f"\nSaved AUROC table ({len(df)} rows) to {args.output}")
    if not df.empty:
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
