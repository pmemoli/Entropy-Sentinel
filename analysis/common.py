import glob
import gzip
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("ES_DATA", os.path.join(REPO, "src", "data"))
RUNS_2048 = os.environ.get("ES_RUNS_2048", os.path.join(DATA, "runs.old.2048"))
RESULTS = os.path.join(REPO, "analysis", "results")
OUT = os.path.join(REPO, "analysis", "out")
TABLES = os.path.join(OUT, "tables")
FIGURES = os.path.join(REPO, "analysis", "out", "figures")
for d in (RESULTS, TABLES, FIGURES):
    os.makedirs(d, exist_ok=True)

MODELS = ["phi3-3b", "ministral3-3b", "ministral3-8b", "qwen3-4b", "qwen3-8b", "gemma3-4b", "gemma3-12b", "llama3-8b", "oss-20b"]
BENCH = ["gsm", "svamp", "gsmsymbolic", "mathhendrycks", "theoremqa", "scibench", "matscibench", "olympiadbench", "livemathbench", "gpqa"]
EXTREMES = ("gsm", "olympiadbench")
MIDRANGE = ("mathhendrycks", "scibench")
CAP = 16000

STATS = ["mean", "std", "max", "q10", "q25", "q50", "q75", "q90", "skew", "kurt", "SEA", "NLLavg", "NLLmax", "NLLsum"]
TEX_STATS = ["Mean", "Std", "Max", "Q10", "Q25", "Q50", "Q75", "Q90", "Skewness", "Kurtosis", "SEA",
             r"$\text{NLL}_{\text{avg}}$", r"$\text{NLL}_{\text{max}}$", r"$\text{NLL}_{\text{sum}}$"]
TEX_MODEL = {"phi3-3b": r"\textsc{Phi-3.5-Mini} (3.8B)", "ministral3-3b": r"\textsc{Ministral3} (3B)", "ministral3-8b": r"\textsc{Ministral3} (8B)",
             "qwen3-4b": r"\textsc{Qwen3} (4B)", "qwen3-8b": r"\textsc{Qwen3} (8B)", "gemma3-4b": r"\textsc{Gemma3} (4B)",
             "gemma3-12b": r"\textsc{Gemma3} (12B)", "llama3-8b": r"\textsc{Llama-3.1} (8B)", "oss-20b": r"\textsc{GPT-OSS} (20B)"}
SHORT_MODEL = {"phi3-3b": "Phi-3.5-Mini 3.8B", "ministral3-3b": "Ministral3 3B", "ministral3-8b": "Ministral3 8B", "qwen3-4b": "Qwen3 4B",
               "qwen3-8b": "Qwen3 8B", "gemma3-4b": "Gemma3 4B", "gemma3-12b": "Gemma3 12B", "llama3-8b": "Llama-3.1 8B", "oss-20b": "GPT-OSS 20B"}


def load(model, bench, data=DATA):
    f = torch.load(f"{data}/features/{model}-{bench}-test.pt", weights_only=False)
    X = np.stack([x["features:"].numpy() for x in f]).astype(float)[:, :14]
    y = np.array([bool(x["success"]) for x in f])
    return X, y


def length(X):
    return np.rint(X[:, 10] / np.maximum(X[:, 0], 1e-9))


def split(model, rng=None):
    rng = np.random.default_rng(42) if rng is None else rng
    d = {}
    for b in BENCH:
        X, y = load(model, b)
        perm = rng.permutation(len(y))
        ntr = int(0.8 * len(y))
        d[b] = dict(X=X, y=y, tr=perm[:ntr], ca=perm[ntr:], L=length(X))
    return d


def pooled(d, group, part):
    return np.vstack([d[b]["X"][d[b][part]] for b in group]), np.concatenate([d[b]["y"][d[b][part]] for b in group])


def gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, class_weight="balanced", random_state=42)


def thr(score, y):
    return np.quantile(score, 1 - y.mean())


def feats(H, lp):
    H = np.asarray(H, float)
    lp = np.asarray(lp, float)
    return np.array([H.mean(), H.std(), H.max(), *np.quantile(H, [.1, .25, .5, .75, .9]), skew(H), kurtosis(H), H.sum(), -lp.mean(), -lp.min(), -lp.sum()])


def read_jsonl(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return [json.loads(line) for line in f]


def results(pattern):
    import pandas as pd
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RESULTS}/{pattern}"))], ignore_index=True)


def miqr(x):
    return f"\\miqr{{{x.median():.3f}}}{{{x.quantile(.75) - x.quantile(.25):.2f}}}"


def write_table(name, rows):
    with open(f"{TABLES}/{name}.tex", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"--- {name}")
    print("\n".join(rows))
