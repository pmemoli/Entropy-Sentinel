"""Scatter of slice AEE vs per-instance AUROC, one point per (LLM, benchmark)
per training group. Visualises the divergence between the two metrics:
many points sit at low AEE despite low AUROC (calibration-not-discrimination).

Mirrors the style of Figure 1 / Figure 3 in the paper: paired-category colour
palette (``#e67e22`` / ``#4682B4``), white-bordered markers, and a stats box
in the upper-left of each axes reporting Pearson R and R^2 between the two
metrics across all (LLM, benchmark) points in that panel.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

CSV = "src/analysis/sensibility_predictive_power/per_instance_auroc.csv"
OUT = "src/analysis/sensibility_predictive_power/aee_vs_auroc_scatter.pdf"

MATH_BENCHES = {
    "gsm-test",
    "gsmsymbolic-test",
    "svamp-test",
    "mathhendrycks-test",
    "olympiadbench-test",
}

FAMILY_STYLE = {
    "Math-style": {"color": "#e67e22", "marker": "s"},
    "Open-domain / Science": {"color": "#4682B4", "marker": "o"},
}

GROUP_TITLE = {"extremes": "Extremes", "intermediate": "Intermediate"}


def main():
    df = pd.read_csv(CSV)
    df["family"] = df["benchmark"].apply(
        lambda b: "Math-style" if b in MATH_BENCHES else "Open-domain / Science"
    )
    df = df.dropna(subset=["auroc", "aee"])

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 8.6), sharex=True, sharey=True)
    for ax, group in zip(axes, ["extremes", "intermediate"]):
        ax.set_box_aspect(1.0)
        sub = df[df["group"] == group]
        for family, style in FAMILY_STYLE.items():
            s = sub[sub["family"] == family]
            ax.scatter(
                s["auroc"], s["aee"],
                color=style["color"], marker=style["marker"],
                s=90, alpha=0.95,
                edgecolors="white", linewidths=1.2,
                label=family, zorder=4,
            )

        r, _ = pearsonr(sub["auroc"], sub["aee"])
        r2 = r * r

        stats_props = dict(
            boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray"
        )
        ax.text(
            0.05, 0.96, f"Pearson R: {r:.3f}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=stats_props, fontsize=8,
        )
        ax.text(
            0.05, 0.86, f"$R^2$: {r2:.3f}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=stats_props, fontsize=8,
        )

        ax.set_xlabel("Per-instance AUROC", fontsize=9)
        ax.set_title(GROUP_TITLE[group], fontsize=10, pad=6)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.40, 0.95)
        ax.set_ylim(0.0, 0.45)

    for ax in axes:
        ax.set_ylabel(r"Slice AEE $|\hat{A}(D) - A(D)|$", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")

    # Also print pooled stats for the writeup.
    r_all, _ = pearsonr(df["auroc"], df["aee"])
    print(f"Pooled across both groups: Pearson R = {r_all:.3f}, R^2 = {r_all**2:.3f}")
    for group in ["extremes", "intermediate"]:
        sub = df[df["group"] == group]
        r, _ = pearsonr(sub["auroc"], sub["aee"])
        print(f"  {group}: Pearson R = {r:.3f}, R^2 = {r**2:.3f}, n={len(sub)}")


if __name__ == "__main__":
    main()
