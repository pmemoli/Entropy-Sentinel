import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde, kurtosis, skew
from sklearn.metrics import roc_auc_score

from common import BENCH, DATA, RUNS_2048, FIGURES, MODELS, RESULTS, TEX_STATS, load, write_table

APPENDIX_PAIRS = [("ministral3-3b", "qwen3-8b"), ("qwen3-4b", "llama3-8b"), ("gemma3-4b", "gemma3-12b")]
HEADER = {"phi3-3b": "phi-3.5-mini 3.8B", "oss-20b": "gpt-oss 20B", "ministral3-3b": "Ministral3 3B", "qwen3-8b": "Qwen3 8B",
          "qwen3-4b": "Qwen3 4B", "llama3-8b": "Llama-3.1 8B", "gemma3-4b": "Gemma3 4B", "gemma3-12b": "Gemma3 12B"}
SENS_NAMES = ["Mean", "Std", "Max", "Q10", "Q25", "Q50", "Q75", "Q90", "Skew", "Kurt", "SEA",
              r"NLL$_{\text{avg}}$", r"NLL$_{\text{max}}$", r"NLL$_{\text{sum}}$", "LNTP", "MTP", "PPL"]


def auroc_incorrect(X, y):
    return np.array([roc_auc_score(~y, -X[:, j] if j in (8, 9) else X[:, j]) for j in range(14)])


def max_entropy_figure():
    X, y = load("phi3-3b", "mathhendrycks")
    ok, bad = X[y, 2], X[~y, 2]
    auc = roc_auc_score(y, -X[:, 2])
    x = np.linspace(0.4, 3, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, gaussian_kde(ok)(x), alpha=0.5, label="Success (Max Entropy)")
    ax.fill_between(x, gaussian_kde(bad)(x), alpha=0.5, label="Failure (Max Entropy)")
    ax.set_title("Distribution of Profile Max on MATH", fontsize=10)
    ax.set_xlabel("Max Entropy of Profile", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.text(0.02, 0.98, f"AUROC: {auc:.4f}", transform=ax.transAxes, fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.savefig(f"{FIGURES}/MATH_test_max_entropy.png", dpi=100, bbox_inches="tight")


def single_stat_summary():
    A = pd.DataFrame([auroc_incorrect(*load(m, b)) for m in MODELS for b in BENCH], columns=TEX_STATS)
    best = A.idxmax(axis=1).value_counts()
    write_table("tab_single_stat_summary", [f"{n} & {A[n].min():.2f}--{A[n].max():.2f} & {A[n].median():.2f} & {100 * best.get(n, 0) / len(A):.0f}\\% \\\\" for n in TEX_STATS])


def auroc_rows(model):
    names = TEX_STATS + ["LNTP", "MTP", "PPL"]
    cols = []
    for b in ("mathhendrycks", "gsm", "olympiadbench"):
        v = list(auroc_incorrect(*load(model, b)))
        v += [v[11], v[12], v[11]]
        order = np.argsort(-np.array(v[:14]))
        cells = [f"{x:.4f}" for x in v]
        cells[order[0]] = f"\\textbf{{{cells[order[0]]}}}"
        cells[order[1]] = f"\\underline{{{cells[order[1]]}}}"
        cols.append(cells)
    return [f"{n} & " + " & ".join(c[i] for c in cols) for i, n in enumerate(names)]


def entropy_analysis_tables():
    rows = []
    for m in ("phi3-3b", "oss-20b"):
        rows += ["\\midrule", f"\\multicolumn{{4}}{{c}}{{\\textit{{\\textsc{{{HEADER[m]}}}}}}} \\\\", "\\midrule"] + [r + " \\\\" for r in auroc_rows(m)]
    write_table("tab_entropy_analysis", rows[1:])
    rows = []
    for a, b in APPENDIX_PAIRS:
        rows += [f"\\multicolumn{{4}}{{c}}{{\\textit{{\\textsc{{{HEADER[a]}}}}}}} & \\multicolumn{{4}}{{c}}{{\\textit{{\\textsc{{{HEADER[b]}}}}}}} \\\\", "\\midrule"]
        rows += [f"{ra} & {rb} \\\\" for ra, rb in zip(auroc_rows(a), auroc_rows(b))] + ["\\midrule"]
    write_table("tab_entropy_analysis_appendix", rows[:-1])


def profile_metrics(H, lp):
    H = np.asarray(H, np.float32)
    lp = np.asarray(lp, np.float32)
    nll = -lp
    return [H.mean(), H.std(), H.max(), *np.percentile(H, [10, 25, 50, 75, 90]), skew(H), kurtosis(H), H.sum(),
            nll.mean(), nll.max(), nll.sum(), np.exp(lp.mean()), np.exp(lp.min()), np.exp(nll.mean())]


def items(path):
    for f in sorted(glob.glob(f"{path}/*.pt")):
        yield from torch.load(f, map_location="cpu", weights_only=False)


def sensitivity_aurocs(its):
    its = [it for it in its if "success" in it]
    M = np.array([profile_metrics(it["entropy_profile"].float().numpy(), it["selected_logprobs"].float().numpy() if torch.is_tensor(it["selected_logprobs"]) else it["selected_logprobs"]) for it in its], float)
    fail = np.array([not it["success"] for it in its])
    auc = np.array([roc_auc_score(fail, np.nan_to_num(M[:, j], nan=np.nanmean(M[:, j]), posinf=1e10, neginf=-1e10)) for j in range(17)])
    auc[[8, 9, 14, 15]] = 1 - auc[[8, 9, 14, 15]]
    return auc


def sensitivity_table():
    rows = []
    for d in sorted(glob.glob(f"{DATA}/sensitivity/phi3-3b-math-sensitivity-t*-s*")):
        t, s = re.search(r"-t([0-9.]+)-s(\d+)$", d).groups()
        its = list(items(d))
        if t == "0.5" and s == "42":
            prompts = {it["prompt"] for it in its}
            its = [it for it in items(f"{RUNS_2048}/phi3-3b-mathhendrycks-test") if it["prompt"] in prompts]
        rows.append([float(t), int(s), *sensitivity_aurocs(its)])
    S = pd.DataFrame(rows, columns=["T", "seed"] + SENS_NAMES)
    S.to_csv(f"{RESULTS}/sensitivity_auroc.csv", index=False)
    f = lambda v: f"{v:.3f}".lstrip("0")
    g = S.groupby("T")
    write_table("tab_sensitivity_auroc", [f"{n} & " + " & ".join(f"{f(g[n].min()[t])}--{f(g[n].max()[t])}" for t in sorted(S["T"].unique())) + " \\\\" for n in SENS_NAMES])


if __name__ == "__main__":
    max_entropy_figure()
    single_stat_summary()
    entropy_analysis_tables()
    sensitivity_table()
