from itertools import combinations

import numpy as np
import pandas as pd

N_CATEGORIES = 11
N_BAD = 4
Z_95 = 1.959963984540054


def fisher_ci(r, n=N_CATEGORIES):
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - Z_95 * se)), float(np.tanh(z + Z_95 * se))


def null_auroc_distribution(n=N_CATEGORIES, k=N_BAD):
    aurocs = []
    for bad in combinations(range(n), k):
        bad = set(bad)
        wins = sum(1 for b in bad for g in range(n) if g not in bad and b > g)
        aurocs.append(wins / (k * (n - k)))
    return np.array(aurocs)


NULL_AUROCS = null_auroc_distribution()


def exact_permutation_p(auroc):
    return float(np.mean(NULL_AUROCS >= auroc - 1e-12))


def add_significance(df):
    df = df.copy()
    df["n_categories"] = N_CATEGORIES
    cis = df["pearson_r"].apply(fisher_ci)
    df["r_ci_lo"] = [lo for lo, _ in cis]
    df["r_ci_hi"] = [hi for _, hi in cis]
    df["perm_p"] = df["auroc_bad4"].apply(exact_permutation_p)
    df["n_subsets"] = len(NULL_AUROCS)
    return df


def main():
    summary = add_significance(
        pd.read_csv("src/results/monitoring_loco_summary.csv")
    ).sort_values("pearson_r", ascending=False)
    summary[
        ["MODEL", "n_categories", "pearson_r", "pearson_p", "r_ci_lo", "r_ci_hi"]
    ].to_csv("src/results/monitoring_loco_pearson_ci.csv", index=False)

    ablation = add_significance(
        pd.read_csv("src/results/monitoring_length_ablation_table.csv")
    )
    ablation[
        [
            "MODEL",
            "feature_set",
            "n_categories",
            "pearson_r",
            "pearson_p",
            "r_ci_lo",
            "r_ci_hi",
        ]
    ].rename(
        columns={"r_ci_lo": "fisher_ci_lo", "r_ci_hi": "fisher_ci_hi"}
    ).to_csv(
        "src/results/monitoring_loco_ablation_pearson_ci.csv", index=False
    )
    ablation[
        ["MODEL", "feature_set", "n_categories", "auroc_bad4", "perm_p", "n_subsets"]
    ].to_csv("src/results/monitoring_loco_auroc_permutation.csv", index=False)

    print("Table 6: LOCO monitoring results per LLM")
    for _, row in summary.iterrows():
        print(
            f"{row['MODEL']:15s} r={row['pearson_r']:.2f} "
            f"[{row['r_ci_lo']:.2f}, {row['r_ci_hi']:.2f}] "
            f"AUROC_bot4={row['auroc_bad4']:.2f} perm_p={row['perm_p']:.3f}"
        )

    print("\nTable 7: length ablation, Pearson r (Fisher 95% CI)")
    order = summary["MODEL"].tolist()
    for model in order:
        sub = ablation[ablation["MODEL"] == model].set_index("feature_set")
        cells = [
            f"{fs}={sub.loc[fs, 'pearson_r']:.2f} "
            f"[{sub.loc[fs, 'r_ci_lo']:.2f}, {sub.loc[fs, 'r_ci_hi']:.2f}]"
            for fs in ["L", "ES", "ES + L"]
        ]
        print(f"{model:15s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
