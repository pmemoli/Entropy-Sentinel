import hashlib
import json
from functools import lru_cache
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd

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

eval_suites = [
    "gsm-test",
    "mathhendrycks-test",
    "gpqa-test",
    "scibench-test",
    "svamp-test",
    "livemathbench-test",
    "gsmsymbolic-test",
    "theoremqa-test",
    "olympiadbench-test",
    "matscibench-test",
]


# %%
@lru_cache(maxsize=None)
def load_model(hash_val: str):
    clf = joblib.load(f"src/data/models/{hash_val}.joblib")
    scaler = joblib.load(f"src/data/models/{hash_val}_scaler.joblib")
    return clf, scaler


def get_benchmark_estimates_from_model(llm, config):
    """Same contract as get_benchmark_estimates, but loads the trained
    model directly and recomputes real/estimated accuracy per suite."""
    train_benchmarks = sorted(config["train_benchmarks"])
    feature_subset = feature_names[: config["feature_size"]]

    model_config = {
        "llm": llm,
        "benchmarks": train_benchmarks,
        "train_suites": [f"{llm}-{s}" for s in train_benchmarks],
        "classifier": config["classifier"],
        "calibrate": config["calibrate"],
        "balance_classes": config["balance_classes"],
        "features": feature_subset,
    }
    hash_val = hashlib.sha256(
        json.dumps(model_config, sort_keys=True).encode()
    ).hexdigest()[:12]

    clf, scaler = load_model(hash_val)
    feature_indices = [feature_names.index(f) for f in feature_subset]

    real_accs, est_accs = {}, {}
    for suite in eval_suites:
        suite_data = torch.load(
            f"src/data/features/{llm}-{suite}.pt", weights_only=False
        )

        if suite in train_benchmarks:
            split_index = int(0.8 * len(suite_data))
            np.random.seed(42)
            np.random.shuffle(suite_data)
            suite_data = suite_data[split_index:]

        X_raw = torch.stack(
            [item["features:"] for item in suite_data]
        ).numpy()[:, feature_indices]
        y = np.array([item["success"] for item in suite_data])

        probs = clf.predict_proba(scaler.transform(X_raw))[:, 1]
        real_accs[suite] = float(np.mean(y))
        est_accs[suite] = float(np.mean(probs))

    return real_accs, est_accs


def model_config_results_from_model(llm, config):
    """Returns pearson r and mae for a given model configuration,
    computed from the per-suite accuracy estimates."""
    real_accs, est_accs = get_benchmark_estimates_from_model(llm, config)
    real_vals = list(real_accs.values())
    est_vals = list(est_accs.values())

    pearson_r = stats.pearsonr(est_vals, real_vals)[0]
    mae = float(np.mean(np.abs(np.array(real_vals) - np.array(est_vals))))

    return float(pearson_r), mae


# %% RQ1: does the entropy-profile signal support cross-domain accuracy estimation under plausible defaults?

"""
For this RQ we fix a RF with class balancing, isotonic calibration and the 10D statistic feature subset as a sensible result.
"""

extremes_config = {
    "train_benchmarks": ["gsm-test", "olympiadbench-test"],
    "classifier": "random_forest",
    "calibrate": True,
    "balance_classes": True,
    "feature_size": 10,
}

intermediate_config = {
    "train_benchmarks": ["mathhendrycks-test", "scibench-test"],
    "classifier": "random_forest",
    "calibrate": True,
    "balance_classes": True,
    "feature_size": 10,
}

# %% Store & print table with model rho/mae on extremes and intermediate config
llms = [
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


table_rows = []
for llm in llms:
    ext_r, ext_mae = model_config_results_from_model(llm, extremes_config)
    inter_r, inter_mae = model_config_results_from_model(
        llm, intermediate_config
    )

    table_rows.append(
        {
            "llm": llm,
            "extremes_r": ext_r,
            "extremes_mae": ext_mae,
            "intermediate_r": inter_r,
            "intermediate_mae": inter_mae,
        }
    )

model_config_table = pd.DataFrame(table_rows)
model_config_table.to_csv(
    "src/results/stemqa_extremes_intermediate_table.csv", index=False
)

print(model_config_table)


# %% Generate figures for all 9 models for extremes
def suite_display_name(suite: str) -> str:
    return suite.replace("-test", "").replace("mathhendrycks", "MATH").upper()


def llm_display_name(llm: str) -> str:
    return "PHI-3.5-MINI" if llm == "phi3-3b" else llm.upper()


def plot_benchmark_estimation(ax, llm, config, s_train=55, s_ood=70):
    """Clean scatter in the style of the monitoring LOCO grid, colored by
    in-domain vs OOD benchmark."""
    real_accs, est_accs = get_benchmark_estimates_from_model(llm, config)

    suites = list(real_accs.keys())
    real_vals = [real_accs[s] for s in suites]
    est_vals = [est_accs[s] for s in suites]
    train_benchmarks = config["train_benchmarks"]

    pearson_corr = stats.pearsonr(est_vals, real_vals)[0]
    mae = float(np.mean(np.abs(np.array(est_vals) - np.array(real_vals))))

    ax.plot(
        [0, 1], [0, 1], color="#e74c3c", linestyle="--", alpha=0.6, zorder=1
    )

    texts = []
    train_val_plotted = False
    ood_plotted = False
    for suite, x, y in zip(suites, est_vals, real_vals):
        if suite in train_benchmarks:
            ax.scatter(
                x,
                y,
                color="#e67e22",
                marker="s",
                s=s_train,
                label=(
                    "In Domain Benchmarks (Test Split)"
                    if not train_val_plotted
                    else ""
                ),
                zorder=5,
                edgecolors="white",
                linewidths=1.2,
            )
            train_val_plotted = True
        else:
            ax.scatter(
                x,
                y,
                color="#4682B4",
                marker="o",
                s=s_ood,
                label="Unseen / OOD Benchmarks" if not ood_plotted else "",
                zorder=4,
                edgecolors="white",
                linewidths=1.2,
            )
            ood_plotted = True

        texts.append(
            ax.text(
                x, y, suite_display_name(suite), fontsize=9, fontweight="bold"
            )
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.set_title(llm_display_name(llm), fontsize=13, fontweight="bold")

    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.5, 1.5),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5, alpha=0.6),
        force_text=(1.5, 1.5),
    )

    ax.text(
        0.05,
        0.95,
        f"r = {pearson_corr:.3f}\nMAE = {mae:.3f}",
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


fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for ax, llm in zip(axes.ravel(), llms):
    plot_benchmark_estimation(ax, llm, extremes_config)

handles, labels = axes.ravel()[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    fontsize=13,
    bbox_to_anchor=(0.5, 0.95),
)

fig.supxlabel("Estimated Accuracy (Expected)", fontsize=14)
fig.supylabel("Real Accuracy (Ground Truth)", fontsize=14)
fig.suptitle(
    "Cross-Domain Accuracy Estimation — Extremes (GSM8K + OlympiadBench)",
    fontsize=16,
    fontweight="bold",
)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
fig.savefig("src/results/stemqa_extremes_all_models.png", dpi=200)
plt.show()

# %% Figure for PHI 3B, MINISTRAL3 8B and GEMMA 12B in a row
highlight_llms = ["phi3-3b", "ministral3-8b", "gemma3-12b"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

for ax, llm in zip(axes, highlight_llms):
    plot_benchmark_estimation(ax, llm, extremes_config)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.92),
)

fig.supxlabel("Estimated Accuracy (Expected)", fontsize=13)
fig.supylabel("Real Accuracy (Ground Truth)", fontsize=13)
fig.suptitle(
    "Cross-Domain Accuracy Estimation — Extremes (GSM8K + OlympiadBench)",
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.82])
fig.savefig("src/results/stemqa_extremes_highlight_row.png", dpi=200)
plt.show()

# %% Same but only PHI 3.5
fig, ax = plt.subplots(figsize=(7, 7))

plot_benchmark_estimation(
    ax, "phi3-3b", extremes_config, s_train=120, s_ood=120
)
ax.set_xlabel("Estimated Accuracy (Expected)", fontsize=12, labelpad=10)
ax.set_ylabel("Real Accuracy (Ground Truth)", fontsize=12, labelpad=10)

ax.legend(loc="lower right", frameon=True, fontsize=10, shadow=True)

fig.tight_layout()
fig.savefig("src/results/stemqa_extremes_phi.png", dpi=200)
plt.show()

# %% EXTRA RQ: How does (llm, benchmark) AUROC and MAE relate (correlation) + plot in the extremes group?

"""
This is super interesting to me since they are basically uncorrelated (R^2 < 0.005). This tells us that the estimation is very noise at the instance level, but the noise becomes much smaller as the results are averaged.  
"""


MATH_SUITES = {
    "gsm-test",
    "mathhendrycks-test",
    "svamp-test",
    "livemathbench-test",
    "gsmsymbolic-test",
    "theoremqa-test",
    "olympiadbench-test",
}
CATEGORY_COLORS = {"math": "#4682B4", "science": "#e67e22"}
CATEGORY_LABELS = {"math": "Math Benchmarks", "science": "Science Benchmarks"}


def suite_category(suite):
    return "math" if suite in MATH_SUITES else "science"


def get_auroc_ae_pairs(llm, config):
    train_benchmarks = sorted(config["train_benchmarks"])
    feature_subset = feature_names[: config["feature_size"]]

    model_config = {
        "llm": llm,
        "benchmarks": train_benchmarks,
        "train_suites": [f"{llm}-{s}" for s in train_benchmarks],
        "classifier": config["classifier"],
        "calibrate": config["calibrate"],
        "balance_classes": config["balance_classes"],
        "features": feature_subset,
    }
    hash_val = hashlib.sha256(
        json.dumps(model_config, sort_keys=True).encode()
    ).hexdigest()[:12]

    clf, scaler = load_model(hash_val)
    feature_indices = [feature_names.index(f) for f in feature_subset]

    pairs = []
    for suite in eval_suites:
        if suite in train_benchmarks:
            continue

        suite_data = torch.load(
            f"src/data/features/{llm}-{suite}.pt", weights_only=False
        )
        X_raw = torch.stack(
            [item["features:"] for item in suite_data]
        ).numpy()[:, feature_indices]
        y = np.array([item["success"] for item in suite_data])

        if len(np.unique(y)) < 2:
            continue

        probs = clf.predict_proba(scaler.transform(X_raw))[:, 1]

        auroc = roc_auc_score(y, probs)
        ae = abs(float(np.mean(probs)) - float(np.mean(y)))

        pairs.append(
            {
                "llm": llm,
                "suite": suite,
                "category": suite_category(suite),
                "auroc": auroc,
                "ae": ae,
            }
        )

    return pairs


def plot_auroc_vs_ae(ax, config, title, csv_path):
    rows = []
    for llm in llms:
        rows.extend(get_auroc_ae_pairs(llm, config))

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    aurocs = df["auroc"].to_numpy()
    aes = df["ae"].to_numpy()
    pearson_r = stats.pearsonr(aurocs, aes)[0]
    pearson_r2 = pearson_r**2
    print(f"{title}: r={pearson_r:.4f}, R^2={pearson_r2:.4f}, n={len(df)}")

    for category, color in CATEGORY_COLORS.items():
        sub = df[df["category"] == category]
        ax.scatter(
            sub["auroc"],
            sub["ae"],
            color=color,
            marker="o",
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=1,
            label=CATEGORY_LABELS[category],
            zorder=3,
        )

    ax.set_xlabel("Per-Instance AUROC", fontsize=12, labelpad=10)
    ax.set_ylabel("Slice MAE", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")

    ax.text(
        0.05,
        0.96,
        f"Pearson $R^2$: {pearson_r2:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            edgecolor="lightgray",
        ),
        fontsize=11,
    )


fig, axes = plt.subplots(2, 1, figsize=(7, 13))

plot_auroc_vs_ae(
    axes[0],
    extremes_config,
    "Slice MAE vs. Per-Instance AUROC — Extremes",
    "src/results/stemqa_extremes_auroc_vs_ae.csv",
)
plot_auroc_vs_ae(
    axes[1],
    intermediate_config,
    "Slice MAE vs. Per-Instance AUROC — Intermediate",
    "src/results/stemqa_intermediate_auroc_vs_ae.csv",
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.97),
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("src/results/stemqa_auroc_vs_ae_column.png", dpi=200)
plt.show()

# %% RQ2: How does this compare to the ATC baseline

"""
We should fix the RQ1 configuration, and compare ES over all our training compositoins to ATC on the SAME training compositions. Table should report MAE median+iqr across training compositions.

This is done since both ES and ATC require some calibration data, so this way they are comparable across a wide range of compositoins.
"""

from itertools import combinations

train_suites_all = [
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

RQ1_CONFIG = {
    "classifier": "random_forest",
    "calibrate": True,
    "balance_classes": True,
    "feature_size": 10,
}

benchmark_groups = [
    list(c) for k in range(1, 5) for c in combinations(train_suites_all, k)
]

atc_cache = {}
for _llm in llms:
    for _suite in eval_suites:
        atc_cache[f"{_llm}-{_suite}"] = torch.load(
            f"src/data/features/{_llm}-{_suite}.pt", weights_only=False
        )

es_mae_csv = pd.read_csv("src/results/stem_classifier_evaluation_results.csv")
es_mae_by_hash = dict(zip(es_mae_csv["hash"], es_mae_csv["mae"]))


def rq1_hash(llm, benchmarks):
    b = sorted(benchmarks)
    cfg = {
        "llm": llm,
        "benchmarks": b,
        "train_suites": [f"{llm}-{s}" for s in b],
        "classifier": RQ1_CONFIG["classifier"],
        "calibrate": RQ1_CONFIG["calibrate"],
        "balance_classes": RQ1_CONFIG["balance_classes"],
        "features": feature_names[: RQ1_CONFIG["feature_size"]],
    }
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]


def suite_split(llm, suite, which):
    data = list(atc_cache[f"{llm}-{suite}"])
    if which in ("train", "test"):
        idx = int(0.8 * len(data))
        np.random.seed(42)
        np.random.shuffle(data)
        data = data[:idx] if which == "train" else data[idx:]
    X = torch.stack([it["features:"] for it in data]).float().numpy()
    y = np.array([it["success"] for it in data])
    return X, y


def atc_all_features_mae(llm, benchmarks, atc_features=None):
    """ATC MAE over all eval suites, once per feature used as the confidence
    score. Each feature is oriented (sign) by its source-set association with
    correctness, then thresholded on the in-domain (source) split so the
    fraction of source scores below t equals the source error; each suite's
    accuracy is estimated as the fraction of scores at or above t (in-domain
    on the held-out split, OOD on the full suite).

    atc_features restricts the candidate confidence scores (default: all 17).
    Pass the RQ1 feature subset to hold ATC to the same features ES sees."""
    if atc_features is None:
        atc_features = feature_names
    train_benchmarks = sorted(benchmarks)

    src_X, src_y = [], []
    for suite in train_benchmarks:
        X, y = suite_split(llm, suite, "train")
        src_X.append(X)
        src_y.append(y)
    src_X = np.vstack(src_X)
    src_y = np.concatenate(src_y)
    source_error = 1.0 - src_y.mean()

    eval_data = []
    for suite in eval_suites:
        which = "test" if suite in train_benchmarks else "full"
        X, y = suite_split(llm, suite, which)
        eval_data.append((X, float(y.mean())))

    maes = {}
    for fname in atc_features:
        fi = feature_names.index(fname)
        s_src = src_X[:, fi]
        if src_y.min() != src_y.max():
            pos = s_src[src_y == 1].mean()
            neg = s_src[src_y == 0].mean()
            sign = 1.0 if pos >= neg else -1.0
        else:
            sign = 1.0

        threshold = np.quantile(sign * s_src, source_error)

        errors = [
            abs(float((sign * X[:, fi] >= threshold).mean()) - real)
            for X, real in eval_data
        ]
        maes[fname] = float(np.mean(errors))

    return maes


def median_iqr(series):
    q25, q50, q75 = series.quantile([0.25, 0.5, 0.75])
    return q50, q75 - q25


# %% RQ2 deployment comparison: ES vs ATC on the Extremes + Intermediate configs
"""
Same two deployment scenarios as RQ1 (Extremes, Intermediate), fixed RQ1
config. ATC is held to the SAME feature subset ES uses (feature_size), so both
methods see identical features and calibrate on the same in-domain source; ATC
just thresholds each feature instead of learning an RF. Reported per model
(9 points, so we count wins rather than quoting an IQR).
"""

deployment_configs = {
    "extremes": extremes_config,
    "intermediate": intermediate_config,
}

deployment_records = []
for dep_name, config in deployment_configs.items():
    atc_features = feature_names[: config["feature_size"]]
    for llm in llms:
        _, es_mae = model_config_results_from_model(llm, config)
        record = {"deployment": dep_name, "llm": llm, "es_mae": es_mae}
        atc_maes = atc_all_features_mae(
            llm, config["train_benchmarks"], atc_features
        )
        for fname, mae in atc_maes.items():
            record[f"atc_{fname}"] = mae
        deployment_records.append(record)

deployment_df = pd.DataFrame(deployment_records)
atc_feat_cols = [c for c in deployment_df.columns if c.startswith("atc_")]
deployment_df["best_atc"] = deployment_df[atc_feat_cols].min(axis=1)
deployment_df["best_atc_feat"] = (
    deployment_df[atc_feat_cols].idxmin(axis=1).str.replace("atc_", "")
)
deployment_df.to_csv(
    "src/results/stemqa_es_vs_atc_deployment_per_model.csv", index=False
)

for dep_name in deployment_configs:
    sub = deployment_df[deployment_df["deployment"] == dep_name]
    n_max = int((sub["es_mae"] < sub["atc_max"]).sum())
    n_best = int((sub["es_mae"] < sub["best_atc"]).sum())
    print(
        f"\n=== {dep_name} (ATC on RQ1 {config['feature_size']}-feature "
        f"subset) ===\n"
        f"ES beats ATC[max]: {n_max}/{len(sub)}   "
        f"ES beats best-ATC: {n_best}/{len(sub)}"
    )
    print(
        sub.set_index("llm")[
            ["es_mae", "atc_max", "best_atc", "best_atc_feat"]
        ]
        .round(4)
        .to_string()
    )


# %% RQ2 full sweep: ES vs ATC across all training compositions
es_atc_records = []
for llm in llms:
    for benchmarks in benchmark_groups:
        h = rq1_hash(llm, benchmarks)
        if h not in es_mae_by_hash:
            continue

        record = {
            "llm": llm,
            "composition": "+".join(sorted(benchmarks)),
            "es_mae": float(es_mae_by_hash[h]),
        }
        for fname, mae in atc_all_features_mae(llm, benchmarks).items():
            record[f"atc_{fname}"] = mae
        es_atc_records.append(record)
    print(f"{llm}: done ({len(benchmark_groups)} compositions)")

es_atc_df = pd.DataFrame(es_atc_records)
es_atc_df.to_csv(
    "src/results/stemqa_es_vs_atc_per_composition.csv", index=False
)


# Median + IQR of MAE across all training compositions (pooled over models).
def mae_stats(series):
    med, iqr = median_iqr(series)
    return {
        "mae_median": med,
        "mae_iqr": iqr,
        "mae_q01": float(series.quantile(0.01)),
        "mae_q02": float(series.quantile(0.02)),
        "mae_q03": float(series.quantile(0.03)),
        "mae_q04": float(series.quantile(0.04)),
        "mae_q05": float(series.quantile(0.05)),
        "mae_q10": float(series.quantile(0.10)),
        "mae_q25": float(series.quantile(0.25)),
        "mae_min": float(series.min()),
    }


summary_rows = [{"method": "ES", **mae_stats(es_atc_df["es_mae"])}]
for fname in feature_names:
    summary_rows.append(
        {"method": f"ATC[{fname}]", **mae_stats(es_atc_df[f"atc_{fname}"])}
    )

es_vs_atc_table = (
    pd.DataFrame(summary_rows).sort_values("mae_median").reset_index(drop=True)
)
es_vs_atc_table["n_compositions"] = len(es_atc_df)
es_vs_atc_table.to_csv("src/results/stemqa_es_vs_atc_mae.csv", index=False)
print(es_vs_atc_table.round(4).to_string(index=False))


# %% RQ3: How sensitive is ES to training composition
"""
Difficulty balance explains much of the composition effect. Under the RQ1
config, for size-k=3 training groups, we summarize each group by its weighted
average accuracy (pooled instance-level success over the group's benchmarks,
per LLM) and relate it to held-out MAE, aggregated over all nine LLMs. Binned
median + IQR reveal a U-shape: intermediate-difficulty groups estimate best.
"""

def rq3_difficulty_figure(k, out_path, csv_path=None):
    points = []
    for llm in llms:
        for benchmarks in benchmark_groups:
            if len(benchmarks) != k:
                continue
            h = rq1_hash(llm, benchmarks)
            if h not in es_mae_by_hash:
                continue
            group_y = np.concatenate(
                [suite_split(llm, s, "train")[1] for s in benchmarks]
            )
            points.append(
                {
                    "llm": llm,
                    "composition": "+".join(sorted(benchmarks)),
                    "weighted_acc": float(group_y.mean()),
                    "mae": float(es_mae_by_hash[h]),
                }
            )

    df = pd.DataFrame(points)
    if csv_path is not None:
        df.to_csv(csv_path, index=False)

    bin_edges = np.arange(0.0, 1.0001, 0.05)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df["bin"] = pd.cut(
        df["weighted_acc"], bin_edges, labels=False, include_lowest=True
    )

    xs, med, lo, hi = [], [], [], []
    for b in range(len(bin_centers)):
        sub = df.loc[df["bin"] == b, "mae"]
        if len(sub) < 10:
            continue
        xs.append(bin_centers[b])
        med.append(sub.median())
        lo.append(sub.quantile(0.25))
        hi.append(sub.quantile(0.75))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        df["weighted_acc"],
        df["mae"],
        s=10,
        color="#4682B4",
        alpha=0.15,
        edgecolors="none",
        zorder=1,
    )
    ax.fill_between(
        xs, lo, hi, color="#4682B4", alpha=0.25, zorder=2, label="IQR"
    )
    ax.plot(
        xs,
        med,
        color="#1f4e79",
        marker="o",
        markersize=5,
        linewidth=2,
        zorder=3,
        label="Median MAE",
    )
    ax.set_xlabel(
        "Training-group weighted accuracy", fontsize=12, labelpad=8
    )
    ax.set_ylabel("Held-out MAE", fontsize=12, labelpad=8)
    ax.set_title(
        f"Estimation quality vs. training-group difficulty (k={k})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(df["weighted_acc"].min(), df["weighted_acc"].max())
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    return df


rq3_difficulty_figure(
    3,
    "src/results/all_llm_accuracy.png",
    "src/results/stemqa_rq3_difficulty_vs_mae.csv",
)
rq3_difficulty_figure(2, "src/results/all_llm_accuracy_k2.png")
rq3_difficulty_figure(4, "src/results/all_llm_accuracy_k4.png")


# %% RQ4: How sensitive is ES to estimator design
"""
Estimator-design main effects: median MAE and Spearman rho (IQR as spread),
marginalized over all training groups and models. Each factor level pools every
sweep row sharing that level, aggregating over the other design axes.
"""

ablation_df = pd.read_csv(
    "src/results/stem_classifier_evaluation_results.csv"
)
ablation_df["feature_dim"] = ablation_df["features"].str.count(" ") + 1


def med_iqr(s):
    q25, q50, q75 = s.quantile([0.25, 0.5, 0.75])
    return q50, q75 - q25


def ablation_rows(factor, column, levels):
    rows = []
    for level, label in levels:
        sub = ablation_df[ablation_df[column] == level]
        mae_med, mae_iqr = med_iqr(sub["mae"])
        rho_med, rho_iqr = med_iqr(sub["spearman_rho"])
        rows.append(
            {
                "factor": factor,
                "setting": label,
                "mae_median": mae_med,
                "mae_iqr": mae_iqr,
                "rho_median": rho_med,
                "rho_iqr": rho_iqr,
                "n": len(sub),
            }
        )
    return rows


ablation_records = (
    ablation_rows(
        "Classifier",
        "classifier",
        [
            ("random_forest", "RF"),
            ("logistic_regression", "LR"),
            ("neural_network", "MLP"),
        ],
    )
    + ablation_rows("Calibration", "calibrate", [(True, "Y"), (False, "N")])
    + ablation_rows(
        "Balancing", "balance_classes", [(True, "Y"), (False, "N")]
    )
    + ablation_rows(
        "Features",
        "feature_dim",
        [(17, "17D"), (10, "10D"), (3, "3D"), (1, "1D")],
    )
)

ablation_table = pd.DataFrame(ablation_records)
ablation_table.to_csv(
    "src/results/stemqa_rq4_ablation_table.csv", index=False
)
print(ablation_table.round(4).to_string(index=False))


def _cell(m, iqr, bold=False):
    num = f"{m:.2f}".lstrip("0")
    if bold:
        num = "\\textbf{" + num + "}"
    return num + "\\textsubscript{" + f"{iqr:.2f}".lstrip("0") + "}"


R = {(r["factor"], r["setting"]): r for r in ablation_records}
best_clf = min(["RF", "LR", "MLP"], key=lambda s: R[("Classifier", s)]["mae_median"])
best_feat_mae = min(
    ["17D", "10D", "3D", "1D"], key=lambda s: R[("Features", s)]["mae_median"]
)
best_feat_rho = max(
    ["17D", "10D", "3D", "1D"], key=lambda s: R[("Features", s)]["rho_median"]
)


def _mae(f, s):
    r = R[(f, s)]
    bold = (f == "Classifier" and s == best_clf) or (
        f == "Features" and s == best_feat_mae
    )
    return _cell(r["mae_median"], r["mae_iqr"], bold)


def _rho(f, s):
    r = R[(f, s)]
    bold = f == "Features" and s == best_feat_rho
    return _cell(r["rho_median"], r["rho_iqr"], bold)


latex = [
    "\\begin{tabular}{llcc}",
    "\\toprule",
    "\\textbf{Factor} & \\textbf{Setting} & \\textbf{MAE} & \\textbf{$\\rho$} \\\\",
    "\\midrule",
    f"\\multirow{{3}}{{*}}{{Classifier}} & RF & {_mae('Classifier','RF')} & {_rho('Classifier','RF')} \\\\",
    f"& LR & {_mae('Classifier','LR')} & {_rho('Classifier','LR')} \\\\",
    f"& MLP & {_mae('Classifier','MLP')} & {_rho('Classifier','MLP')} \\\\",
    "\\midrule",
    f"Calibration & Y / N & {_mae('Calibration','Y')} / {_mae('Calibration','N')} & {_rho('Calibration','Y')} / {_rho('Calibration','N')} \\\\",
    f"Balancing & Y / N & {_mae('Balancing','Y')} / {_mae('Balancing','N')} & {_rho('Balancing','Y')} / {_rho('Balancing','N')} \\\\",
    "\\midrule",
    f"\\multirow{{4}}{{*}}{{Features}} & 17D & {_mae('Features','17D')} & {_rho('Features','17D')} \\\\",
    f"& 10D & {_mae('Features','10D')} & {_rho('Features','10D')} \\\\",
    f"& 3D & {_mae('Features','3D')} & {_rho('Features','3D')} \\\\",
    f"& 1D & {_mae('Features','1D')} & {_rho('Features','1D')} \\\\",
    "\\bottomrule",
    "\\end{tabular}",
]

with open("src/results/stemqa_rq4_ablation_table.tex", "w") as f:
    f.write("\n".join(latex) + "\n")
print("\n".join(latex))
