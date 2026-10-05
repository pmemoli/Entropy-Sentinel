import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

from src.analysis.metric_predictive_power.metric_power import iter_samples_from_dir

OUT = "src/results/phi3_math_max_entropy_density.png"
FONT_SIZE = 10


def max_entropies(suite_dir):
    success, failure = [], []
    for item in iter_samples_from_dir(suite_dir):
        value = torch.max(item["entropy_profile"].to(torch.float32)).item()
        (success if item["success"] else failure).append(value)
    return success, failure


def plot(success, failure, out):
    labels = [1] * len(success) + [0] * len(failure)
    auroc = roc_auc_score(labels, [-e for e in success + failure])

    x = np.linspace(0.4, 3, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        x, gaussian_kde(success)(x), alpha=0.5, label="Success (Max Entropy)"
    )
    ax.fill_between(
        x, gaussian_kde(failure)(x), alpha=0.5, label="Failure (Max Entropy)"
    )
    ax.set_title("Distribution of Profile Max on MATH", fontsize=FONT_SIZE)
    ax.set_xlabel("Max Entropy of Profile", fontsize=FONT_SIZE)
    ax.set_ylabel("Density", fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, loc="upper right")
    ax.text(
        0.02,
        0.98,
        f"AUROC: {auroc:.4f}",
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.savefig(out, dpi=100, bbox_inches="tight")
    return auroc


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs_root", default="src/data/runs")
    p.add_argument("--suite", default="phi3-3b-mathhendrycks-test")
    p.add_argument("--output", default=OUT)
    args = p.parse_args()

    success, failure = max_entropies(os.path.join(args.runs_root, args.suite))
    auroc = plot(success, failure, args.output)
    print(f"n={len(success) + len(failure)} AUROC={auroc:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
