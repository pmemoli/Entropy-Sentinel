import pandas as pd

SENSITIVITY_CSV = "src/analysis/sensibility_predictive_power/sensitivity_auroc.csv"
OUT_CSV = "src/results/temp_sensitivity_auroc_ranges.csv"

METRICS = [
    ("Mean", "Mean"),
    ("STD", "Std"),
    ("Max", "Max"),
    ("Q10", "Q10"),
    ("Q25", "Q25"),
    ("Q50", "Q50"),
    ("Q75", "Q75"),
    ("Q90", "Q90"),
    ("Skew", "Skew"),
    ("Kurt", "Kurt"),
    ("SE_sum", "SEA"),
    ("NLL_avg", "NLLavg"),
    ("NLL_max", "NLLmax"),
    ("NLL_sum", "NLLsum"),
    ("LNTP", "LNTP"),
    ("MTP", "MTP"),
    ("PPL", "PPL"),
]


def _fmt(x):
    return f"{x:.3f}".lstrip("0")


def main():
    df = pd.read_csv(SENSITIVITY_CSV)
    if "source" in df.columns:
        og = df[(df["temperature"] == 0.5) & (df["seed"] == 42)]
        assert (og["source"] == "original").all(), (
            "T=0.5 seed 42 must use the original paper generations; rerun "
            "sensitivity_auroc.py with --original_t05_dir"
        )

    temperatures = sorted(df["temperature"].unique())
    rows = []
    for col, label in METRICS:
        row = {"stat": label}
        for t in temperatures:
            vals = df.loc[df["temperature"] == t, col]
            row[f"t{t}_min"] = vals.min()
            row[f"t{t}_max"] = vals.max()
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)

    print("Table 9: per-statistic AUROC ranges (min-max across seeds)")
    print(f"{'Stat':8s}" + "".join(f"{'T = ' + str(t):>14s}" for t in temperatures))
    for _, row in table.iterrows():
        cells = [
            f"{_fmt(row[f't{t}_min'])}-{_fmt(row[f't{t}_max'])}"
            for t in temperatures
        ]
        print(f"{row['stat']:8s}" + "".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
