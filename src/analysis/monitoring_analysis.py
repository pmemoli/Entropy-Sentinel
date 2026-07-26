"""
We show spearman (and aee?) median-iqr for k = 1...7 and have a final LOCO (leave one category out) row.

Results saved for each model.
"""

import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from itertools import combinations
import pandas as pd


# %% Main function to train and evaluate the model on OOD categories
"""
RQ1: Can Entropy Sentinel distilled from a judge monitor OOD categories?

We report LOCO (leave one category out) evaluation, where we train on all categories except one and evaluate on the left-out category. We report spearman rho and MAE for each left-out category.

The idea is to see if the model can generalize to unseen categories and if it can capture the underlying distribution of judge scores across different categories. We hypothesize that there is something about the entropy signature that is generalizable across categories, despite them being subjective.
"""


# %%
ALL_SUITES = [
    "llama3-8b-wildbench-test",
    "gemma3-4b-wildbench-test",
    "gemma3-12b-wildbench-test",
    "phi3-3b-wildbench-test",
    "ministral3-3b-wildbench-test",
    "ministral3-8b-wildbench-test",
    "oss-20b-wildbench-test",
    "qwen3-4b-wildbench-test",
    "qwen3-8b-wildbench-test",
]

suite_X = {}
suite_y = {}
suite_cat = {}
suite_combinations = {}

for SUITE in ALL_SUITES:
    features_path = f"src/data/features/{SUITE}.pt"

    if not os.path.exists(features_path):
        print(f"Skipping {SUITE}: no features at {features_path}")
        continue

    data = torch.load(features_path)

    suite_X[SUITE] = np.array([row["features:"] for row in data])
    suite_y[SUITE] = np.array([row["judge_score"] for row in data])
    suite_cat[SUITE] = np.array([row["primary_category"] for row in data])

    categories = list(set(suite_cat[SUITE]))
    suite_combinations[SUITE] = {
        k: list(combinations(categories, k)) for k in range(1, 6)
    }

    print(f"{SUITE}: {len(data)} responses, {len(categories)} categories")

SUITES = [suite for suite in ALL_SUITES if suite in suite_X]


# %% Feature <-> category pearson R table
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

feat_cat_rows = []
for SUITE in SUITES:
    X = suite_X[SUITE]
    y = suite_y[SUITE]
    cat = suite_cat[SUITE]

    categories = sorted(np.unique(cat))
    cat_avg_score = np.array([y[cat == c].mean() for c in categories])

    row = {"MODEL": SUITE.replace("-wildbench-test", "")}
    for fi, fname in enumerate(feature_names):
        cat_avg_feat = np.array(
            [X[cat == c, fi].mean() for c in categories]
        )
        row[fname] = pearsonr(cat_avg_feat, cat_avg_score)[0]
    feat_cat_rows.append(row)

feature_category_pearson = pd.DataFrame(feat_cat_rows).set_index("MODEL")
feature_category_pearson.to_csv(
    "src/results/monitoring_feature_category_pearson.csv"
)
print(feature_category_pearson.round(3).to_string())


# %% LOCO (leave one category out) evaluation
def category_balanced_weights(cat_arr):
    """Weight each response inversely to its category size so every category
    contributes equally regardless of how many responses it has."""
    counts = {c: (cat_arr == c).sum() for c in np.unique(cat_arr)}
    w = np.array([1.0 / counts[c] for c in cat_arr])
    return w * len(w) / w.sum()


loco_results = {}

for SUITE in SUITES:
    print(f"===== {SUITE}")

    X = suite_X[SUITE]
    y = suite_y[SUITE]
    cat = suite_cat[SUITE]

    OOD_real_scores = {c: y[cat == c].mean() for c in np.unique(cat)}
    OOD_estimate_scores = {}
    OOD_centered_scores = {}

    for ood_cat in np.unique(cat):
        print("Training for OOD category:", ood_cat)

        X_train = X[cat != ood_cat]
        y_train = y[cat != ood_cat]
        cat_train = cat[cat != ood_cat]

        X_test = X[cat == ood_cat]

        y_train = (y_train - 1) / 9

        # Fixed config: all 17 features, depth 8, category-balanced weights,
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=category_balanced_weights(cat_train),
        )

        pred_test = model.predict(X_test).mean()
        pred_train = model.predict(X_train).mean()
        OOD_estimate_scores[ood_cat] = pred_test * 9 + 1
        OOD_centered_scores[ood_cat] = pred_test - pred_train

    # Ranking uses the bias-corrected estimate; MAE uses the calibrated one.
    pearson_r, pearson_p = pearsonr(
        list(OOD_real_scores.values()), list(OOD_centered_scores.values())
    )

    mae = float(
        np.mean(
            [
                abs(OOD_real_scores[c] - OOD_estimate_scores[c])
                for c in OOD_real_scores
            ]
        )
    )

    # AUROC for flagging the 4 worst-performing categories (by true score),
    # ranked by the bias-corrected estimate.
    real_arr = np.array(list(OOD_real_scores.values()))
    cen_arr = np.array(list(OOD_centered_scores.values()))
    is_bad = np.zeros(len(real_arr), dtype=int)
    is_bad[np.argsort(real_arr)[:4]] = 1
    auroc_bad4 = float(roc_auc_score(is_bad, -cen_arr))

    loco_results[SUITE] = {
        "real_scores": {str(c): float(v) for c, v in OOD_real_scores.items()},
        "estimate_scores": {
            str(c): float(v) for c, v in OOD_estimate_scores.items()
        },
        "centered_scores": {
            str(c): float(v) for c, v in OOD_centered_scores.items()
        },
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mae": mae,
        "auroc_bad4": auroc_bad4,
    }

    print(
        f"LOCO Pearson r: {pearson_r:.3f} (p={pearson_p:.3f}), "
        f"AUROC(bad4): {auroc_bad4:.3f}, MAE: {mae:.3f}"
    )

# %% Store LOCO results
os.makedirs("src/results", exist_ok=True)

with open("src/results/monitoring_loco_results.json", "w") as f:
    json.dump(loco_results, f, indent=4)

loco_summary = pd.DataFrame(
    [
        {
            "MODEL": suite.replace("-wildbench-test", ""),
            "pearson_r": loco_results[suite]["pearson_r"],
            "pearson_p": loco_results[suite]["pearson_p"],
            "auroc_bad4": loco_results[suite]["auroc_bad4"],
            "mae": loco_results[suite]["mae"],
        }
        for suite in SUITES
    ]
).sort_values("pearson_r", ascending=False)

loco_summary.to_csv("src/results/monitoring_loco_summary.csv", index=False)
print(loco_summary.to_string(index=False))


# %% Graph in same style as STEM exp 1
def prettify(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


for SUITE in SUITES:
    OOD_real_scores = loco_results[SUITE]["real_scores"]
    OOD_centered_scores = loco_results[SUITE]["centered_scores"]

    categories_order = list(OOD_real_scores.keys())
    real_vals = np.array([OOD_real_scores[c] for c in categories_order])
    est_vals = np.array([OOD_centered_scores[c] for c in categories_order])

    # Express both axes relative to the leave-one-out training-group baseline,
    # then standardize (z-score) each so they share a unitless scale: the true
    # score minus the mean true score of the other categories, versus the
    # bias-corrected (train-baseline-relative) estimate. On this scale the
    # 45-degree line is perfect agreement and its visual slope is Pearson r.
    n = len(real_vals)
    real_dev = np.array(
        [real_vals[i] - np.delete(real_vals, i).mean() for i in range(n)]
    )
    real_z = (real_dev - real_dev.mean()) / (real_dev.std() + 1e-9)
    est_z = (est_vals - est_vals.mean()) / (est_vals.std() + 1e-9)

    loco_r = loco_results[SUITE]["pearson_r"]
    loco_p = loco_results[SUITE]["pearson_p"]

    fig, ax = plt.subplots(figsize=(7, 7))

    lim = max(np.abs(real_z).max(), np.abs(est_z).max()) * 1.15
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color="#e74c3c",
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )
    ax.axhline(0, color="gray", linewidth=1, alpha=0.4, zorder=1)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.4, zorder=1)

    ax.scatter(
        real_z,
        est_z,
        color="#4682B4",
        marker="o",
        s=120,
        alpha=1,
        edgecolors="white",
        linewidths=1.5,
        zorder=4,
    )

    for c, xv, yv in zip(categories_order, real_z, est_z):
        ax.text(
            xv + 0.02 * lim,
            yv + 0.02 * lim,
            prettify(c),
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel(
        "True Score − Train Baseline (z-scored)", fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        "Estimate Relative to Train Baseline (z-scored)",
        fontsize=12,
        labelpad=10,
    )
    if SUITE == "phi3-3b-wildbench-test":
        ax.set_title("PHI-3.5-MINI", fontsize=13, fontweight="bold")
    else:
        ax.set_title(
            f"Cross-Category Score Estimation "
            f"({SUITE.replace('-wildbench-test', '').upper()})",
            fontsize=14,
            pad=15,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")

    stats_props = dict(
        boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray"
    )
    ax.text(
        0.05,
        0.96,
        f"Pearson r: {loco_r:.4f} (p={loco_p:.3f})",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=stats_props,
        fontsize=11,
    )

    plt.tight_layout()
    fig.savefig(f"src/results/{SUITE}-loco-ranking.png", dpi=200)
    plt.show()


# %% All models in a 3x3 grid (sorted by Pearson r)
grid_suites = sorted(
    SUITES, key=lambda s: loco_results[s]["pearson_r"], reverse=True
)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for ax, SUITE in zip(axes.ravel(), grid_suites):
    OOD_real_scores = loco_results[SUITE]["real_scores"]
    OOD_centered_scores = loco_results[SUITE]["centered_scores"]

    categories_order = list(OOD_real_scores.keys())
    real_vals = np.array([OOD_real_scores[c] for c in categories_order])
    est_vals = np.array([OOD_centered_scores[c] for c in categories_order])

    n = len(real_vals)
    real_dev = np.array(
        [real_vals[i] - np.delete(real_vals, i).mean() for i in range(n)]
    )
    real_z = (real_dev - real_dev.mean()) / (real_dev.std() + 1e-9)
    est_z = (est_vals - est_vals.mean()) / (est_vals.std() + 1e-9)

    loco_r = loco_results[SUITE]["pearson_r"]
    loco_p = loco_results[SUITE]["pearson_p"]

    lim = max(np.abs(real_z).max(), np.abs(est_z).max()) * 1.15
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color="#e74c3c",
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )
    ax.axhline(0, color="gray", linewidth=1, alpha=0.4, zorder=1)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.4, zorder=1)

    ax.scatter(
        real_z,
        est_z,
        color="#4682B4",
        marker="o",
        s=70,
        alpha=1,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.set_title(
        SUITE.replace("-wildbench-test", ""), fontsize=13, fontweight="bold"
    )
    ax.text(
        0.05,
        0.95,
        f"r = {loco_r:.2f} (p={loco_p:.3f})",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            edgecolor="lightgray",
        ),
        fontsize=10,
    )

fig.supxlabel("True Score − Train Baseline (z-scored)", fontsize=14)
fig.supylabel("Estimate Relative to Train Baseline (z-scored)", fontsize=14)
fig.suptitle(
    "Leave-One-Category-Out category-level quality estimation",
    fontsize=16,
    fontweight="bold",
)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.98])
fig.savefig("src/results/loco_monitoring_grid.png", dpi=200)
plt.show()


# %% Length ablation
"""
Original MTBench paper mentions how length biased the llm as a judge is. To evaluate that the model is not just capturing length (since some features like SEA are very much correlated to it), we'll perform length ablations.
"""

suite_lengths = {}

for SUITE in SUITES:
    tensor_path = f"src/data/runs/{SUITE}"
    tensor_files = os.listdir(tensor_path)

    lengths = []
    judge_scores = []
    for file in tensor_files:
        tensor_data = torch.load(os.path.join(tensor_path, file))
        for tensor_item in tensor_data:
            for message in tensor_item["messages"]:
                if message["role"] != "assistant":
                    continue
                if message.get("judge_score") is None:
                    continue

                lengths.append(len(message["token_ids"]))
                judge_scores.append(message["judge_score"])

    suite_lengths[SUITE] = np.array(lengths, dtype=float)

    print(f"{SUITE}: {len(lengths)} lengths, {len(suite_y[SUITE])} features")

# %% Train length baselines with the same LOCO setup as the main experiment.

"""
Compare three feature sets under the identical LOCO pipeline (all categories
held out one at a time, category-balanced weights, depth-8 RF, bias-corrected
estimate): L (response length alone), ES (the 17 entropy features), and ES + L.
If ES ~ ES + L >> L, the entropy signal is not merely a length proxy.
"""

result_length_ablation_path = (
    "src/results/monitoring_ood_category_length_ablation_results.json"
)


def loco_metrics(X, y, cat):
    """LOCO with the settled config; returns the same metrics as the main
    experiment (bias-corrected Pearson, AUROC over the 4 worst categories,
    calibrated MAE)."""
    real, centered, calibrated = {}, {}, {}
    for ood_cat in np.unique(cat):
        tr = cat != ood_cat
        te = cat == ood_cat
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        model.fit(
            X[tr],
            (y[tr] - 1) / 9,
            sample_weight=category_balanced_weights(cat[tr]),
        )
        pred_test = model.predict(X[te]).mean()
        pred_train = model.predict(X[tr]).mean()
        real[ood_cat] = y[te].mean()
        calibrated[ood_cat] = pred_test * 9 + 1
        centered[ood_cat] = pred_test - pred_train

    real_arr = np.array(list(real.values()))
    cen_arr = np.array(list(centered.values()))
    cal_arr = np.array(list(calibrated.values()))

    pearson_r, pearson_p = pearsonr(real_arr, cen_arr)
    mae = float(np.mean(np.abs(real_arr - cal_arr)))
    is_bad = np.zeros(len(real_arr), dtype=int)
    is_bad[np.argsort(real_arr)[:4]] = 1
    auroc_bad4 = float(roc_auc_score(is_bad, -cen_arr))

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "auroc_bad4": auroc_bad4,
        "mae": mae,
    }


length_ablation_results = {}

for SUITE in SUITES:
    print(f"===== {SUITE}")

    X = suite_X[SUITE]
    y = suite_y[SUITE]
    cat = suite_cat[SUITE]

    L = suite_lengths[SUITE].reshape(-1, 1)
    feature_sets = {"L": L, "ES": X, "ES + L": np.hstack([X, L])}

    length_ablation_results[SUITE] = {}
    for feature_name, feature_X in feature_sets.items():
        res = loco_metrics(feature_X, y, cat)
        length_ablation_results[SUITE][feature_name] = res
        print(
            f"  {feature_name:6s} Pearson r: {res['pearson_r']:.3f} "
            f"(p={res['pearson_p']:.3f}), AUROC(bad4): {res['auroc_bad4']:.3f}, "
            f"MAE: {res['mae']:.3f}"
        )

with open(result_length_ablation_path, "w") as f:
    json.dump(length_ablation_results, f, indent=4)

# %% Length ablation summary table
table_rows = [
    {
        "MODEL": SUITE.replace("-wildbench-test", ""),
        "feature_set": feature_name,
        "pearson_r": res["pearson_r"],
        "pearson_p": res["pearson_p"],
        "auroc_bad4": res["auroc_bad4"],
        "mae": res["mae"],
    }
    for SUITE in SUITES
    for feature_name, res in length_ablation_results[SUITE].items()
]

length_ablation_table = pd.DataFrame(table_rows)
length_ablation_table.to_csv(
    "src/results/monitoring_length_ablation_table.csv", index=False
)
print(length_ablation_table.to_string(index=False))
