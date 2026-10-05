import numpy as np
import pandas as pd
import torch

SWEEP_CSV = "src/results/stem_classifier_evaluation_results.csv"

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

ABBREV = {
    "gsm-test": "GSM",
    "gsmsymbolic-test": "GSMSYM",
    "svamp-test": "SVAMP",
    "mathhendrycks-test": "MATH",
    "olympiadbench-test": "OLY",
    "gpqa-test": "GPQA",
    "scibench-test": "SCI",
    "theoremqa-test": "THM",
    "livemathbench-test": "LIVE",
    "matscibench-test": "MAT",
}


def med_iqr(s):
    q25, q50, q75 = s.quantile([0.25, 0.5, 0.75])
    return q50, q75 - q25


def benchmark_stats(suites):
    acc, size = {}, {}
    for s in suites:
        accs = []
        for llm in LLMS:
            data = torch.load(f"src/data/features/{llm}-{s}.pt", weights_only=False)
            accs.append(np.mean([item["success"] for item in data]))
        acc[s] = float(np.mean(accs))
        size[s] = len(data)
    return acc, size


def group_accuracy(benchmarks, acc, size):
    total = sum(size[b] for b in benchmarks)
    return sum(acc[b] * size[b] for b in benchmarks) / total


def _cell(m, iqr, digits=2):
    return f"{m:.{digits}f}".lstrip("0") + "_" + f"{iqr:.{digits}f}".lstrip("0")


def main():
    df = pd.read_csv(SWEEP_CSV)
    df["k"] = df["benchmarks"].str.split().str.len()

    k_rows = []
    for llm in LLMS:
        row = {"llm": llm}
        for k in range(1, 5):
            m, iqr = med_iqr(df.loc[(df["llm"] == llm) & (df["k"] == k), "mae"])
            row[f"k{k}_mae_median"] = m
            row[f"k{k}_mae_iqr"] = iqr
        k_rows.append(row)
    k_table = pd.DataFrame(k_rows)
    k_table.to_csv("src/results/stemqa_rq3_mae_vs_k.csv", index=False)

    print("Table 4: median MAE vs k training benchmarks (IQR after _)")
    print(f"{'Model':15s}" + "".join(f"{k:>10d}" for k in range(1, 5)))
    for _, row in k_table.iterrows():
        print(
            f"{row['llm']:15s}"
            + "".join(
                f"{_cell(row[f'k{k}_mae_median'], row[f'k{k}_mae_iqr']):>10s}"
                for k in range(1, 5)
            )
        )

    suites = sorted({s for b in df["benchmarks"] for s in b.split()})
    acc, size = benchmark_stats(suites)

    combo = (
        df.groupby("benchmarks")["mae"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: "q25", 0.5: "mae_median", 0.75: "q75"})
    )
    combo["mae_iqr"] = combo["q75"] - combo["q25"]
    combo["k"] = combo.index.str.split().str.len()

    extreme_rows = []
    for which, ascending in [("best", True), ("worst", False)]:
        for k in range(1, 5):
            sub = combo[combo["k"] == k].sort_values("mae_median", ascending=ascending)
            benchmarks = sub.index[0].split()
            extreme_rows.append(
                {
                    "which": which,
                    "k": k,
                    "benchmarks": " ".join(benchmarks),
                    "mae_median": sub.iloc[0]["mae_median"],
                    "mae_iqr": sub.iloc[0]["mae_iqr"],
                    "group_acc": group_accuracy(benchmarks, acc, size),
                }
            )
    extremes = pd.DataFrame(extreme_rows)
    extremes.to_csv("src/results/stemqa_best_worst_combinations.csv", index=False)

    print("\nTable 11: best and worst benchmark combinations per k")
    for which in ["best", "worst"]:
        print(f"{which.capitalize()} performing")
        for _, row in extremes[extremes["which"] == which].iterrows():
            names = ", ".join(sorted(ABBREV[s] for s in row["benchmarks"].split()))
            print(
                f"  {row['k']}  {names:28s} "
                f"{_cell(row['mae_median'], row['mae_iqr'], 3):>10s} "
                f"{row['group_acc']:.3f}".replace(" 0.", " .")
            )


if __name__ == "__main__":
    main()
