import glob
import itertools
import os
import re
import string

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

from common import BENCH, DATA, RUNS_2048, RESULTS, STATS, TEX_STATS, feats, gbm, load, read_jsonl, thr, write_table

API_BENCH = ["mathhendrycks", "theoremqa", "scibench", "olympiadbench", "gpqa"]
API_NAMES = {"mathhendrycks": "MATH", "theoremqa": "TheoremQA", "scibench": "SciBench", "olympiadbench": "OlympiadBench", "gpqa": "GPQA"}
QA_BENCH = ["triviaqa", "nqopen", "popqa"]
QA_NAMES = {"triviaqa": "TriviaQA", "nqopen": "NQ-open", "popqa": "PopQA"}
TOPK_STATS = ["Mean", "Std", "Max", "Q10", "Q25", "Q50", "Q75", "Q90", "Skew", "Kurt", "SEA"]
PRE, CAL, POST, N, REPS = 40, 20, 10, 300, 50


def stat_aurocs(X, y):
    a = {n: roc_auc_score(y, -X[:, j]) for j, n in enumerate(STATS)}
    a["skew"], a["kurt"] = 1 - a["skew"], 1 - a["kurt"]
    return a


def holdout(d):
    rng = np.random.default_rng(42)
    for b in d:
        perm = rng.permutation(len(d[b]["y"]))
        ntr = int(0.8 * len(perm))
        d[b]["tr"], d[b]["ca"] = perm[:ntr], perm[ntr:]
    return d


def train_estimators(Xtr, ytr, Xca, yca):
    g = gbm().fit(Xtr, ytr)
    j = int(np.argmax([roc_auc_score(yca, -Xca[:, i]) for i in range(14)]))
    T = dict(probe=thr(g.predict_proba(Xca)[:, 1], yca), sel=thr(-Xca[:, j], yca), sea=thr(-Xca[:, 10], yca), nllsum=thr(-Xca[:, 13], yca))

    def estimate(X):
        p = g.predict_proba(X)[:, 1]
        return dict(probe=(p > T["probe"]).mean(), sel=(-X[:, j] > T["sel"]).mean(), sea=(-X[:, 10] > T["sea"]).mean(),
                    nllsum=(-X[:, 13] > T["nllsum"]).mean(), avg=p.mean())
    return estimate, STATS[j]


def within_sweep(d, ks):
    rows = []
    for k in ks:
        for G in itertools.combinations(list(d), k):
            Xtr, ytr = np.vstack([d[b]["X"][d[b]["tr"]] for b in G]), np.concatenate([d[b]["y"][d[b]["tr"]] for b in G])
            Xca, yca = np.vstack([d[b]["X"][d[b]["ca"]] for b in G]), np.concatenate([d[b]["y"][d[b]["ca"]] for b in G])
            if ytr.min() == ytr.max() or yca.min() == yca.max():
                continue
            estimate, _ = train_estimators(Xtr, ytr, Xca, yca)
            for b in [b for b in d if b not in G]:
                e = estimate(d[b]["X"])
                rows.append(dict(k=k, group="+".join(G), held=b, true=d[b]["y"].mean(), **{f"e_{c}": abs(v - d[b]["y"].mean()) for c, v in e.items()}))
    return pd.DataFrame(rows)


def api_tables():
    labels = pd.read_json(f"{DATA}/api/labels_gpt41mini16k.json", typ="series").to_dict()
    recs = [r for b in API_BENCH for r in read_jsonl(f"{DATA}/api/gen_gpt41mini16k_{b}.jsonl.gz") if r["custom_id"] in labels and len(r["entropy_profile"]) >= 2]
    d, rows = {}, []
    for b in API_BENCH:
        rs = [r for r in recs if r["bench"] == b]
        X, y = np.stack([feats(r["entropy_profile"], r["selected_logprobs"]) for r in rs]), np.array([labels[r["custom_id"]] for r in rs])
        d[b] = dict(X=X, y=y)
        a = stat_aurocs(X, y)
        best = max(STATS, key=a.get)
        trunc = 100 * np.mean([r["truncated"] for r in rs])
        rows.append(f"{API_NAMES[b]} & {len(y)} & {y.mean():.2f} & {np.mean([len(r['entropy_profile']) for r in rs]):.0f} & "
                    f"{f'{trunc:.1f}' if 0 < trunc < 1 else f'{trunc:.0f}'} & {a['SEA']:.2f} & {TEX_STATS[STATS.index(best)].replace('Skewness', 'Skew').replace('Kurtosis', 'Kurt')} ({a[best]:.2f}) \\\\")
    write_table("tab_api_stem", rows)
    D = within_sweep(holdout(d), (1, 2, 3))
    acc = {b: d[b]["y"].mean() for b in API_BENCH}
    n = {b: len(d[b]["y"]) for b in API_BENCH}
    D["e_const"] = (D.group.map(lambda g: sum(acc[b] * n[b] for b in g.split("+")) / sum(n[b] for b in g.split("+"))) - D.true).abs()
    G = D.groupby(["k", "group"]).mean(numeric_only=True).reset_index()
    G.to_csv(f"{RESULTS}/api_sweep.csv", index=False)
    rows = []
    for k, h in G.groupby("k"):
        m = h.median(numeric_only=True)
        rows.append(f"{k} & {len(h)} & " + " & ".join(f"{m[f'e_{c}']:.3f}" for c in ("probe", "sel", "sea", "nllsum", "avg", "const")) + f" & {(h.e_probe < h.e_sel).mean():.2f} \\\\")
    write_table("tab_api_sweep", rows)


def boxed_number(t):
    t = re.sub(r"[₣₺₦$€£¥,]", "", t)
    m = re.findall(r"-?\d+\.?\d*", t)
    return m[-1] if m else None


def gsm_correct(rec):
    g = rec["generation"]
    b = re.findall(r"\\boxed\{([^{}]*)\}", g)
    m = re.search(r"[Ff]inal [Aa]nswer[^0-9\-]*(-?[\d,]+\.?\d*)", g)
    gn = boxed_number(b[-1]) if b else (boxed_number(m.group(1)) if m else boxed_number(g))
    rn = boxed_number(rec["reference"].split("####")[-1])
    if rn is None or gn is None:
        return False
    try:
        return abs(float(rn) - float(gn)) < 1e-6
    except ValueError:
        return False


def qa_correct(rec):
    norm = lambda s: " ".join(re.sub(r"\b(a|an|the)\b", " ", "".join(ch for ch in s.lower() if ch not in set(string.punctuation))).split())
    g = rec["generation"] or ""
    b = re.findall(r"\\boxed\{([^{}]*)\}", g)
    c = norm(re.sub(r"\\text\{([^}]*)\}", r"\1", b[-1] if b else (g.strip().split("\n")[-1] if g.strip() else "")))
    return any(a == c or (a and a in c) for a in (norm(a) for a in rec["answers"] if a))


def qwen_stem_estimators(benches, rng):
    Xs, ys = zip(*[load("qwen3-4b", b) for b in benches])
    X, y = np.vstack(Xs), np.concatenate(ys)
    perm = rng.permutation(len(y))
    ntr = int(0.8 * len(y))
    return X[perm[:ntr]], y[perm[:ntr]], X[perm[ntr:]], y[perm[ntr:]]


def pool(path):
    R = read_jsonl(path)
    return dict(X=np.stack([feats(r["entropy_profile"], r["selected_logprobs"]) for r in R]), y=np.array([gsm_correct(r) for r in R]),
                L=np.array([len(r["entropy_profile"]) for r in R], float), n=len(R))


def regression_streams():
    Xtr, ytr, Xca, yca = qwen_stem_estimators(("mathhendrycks", "scibench"), np.random.default_rng(42))
    g = gbm().fit(Xtr, ytr)
    j = int(np.argmax([roc_auc_score(yca, -Xca[:, i]) for i in range(14)]))
    tg, tsel, tnll = thr(g.predict_proba(Xca)[:, 1], yca), thr(-Xca[:, j], yca), thr(-Xca[:, 13], yca)
    base = pool(f"{DATA}/regression/full_base.jsonl.gz")
    tlen = np.quantile(-base["L"], 1 - base["y"].mean())
    est = {"probe": lambda P, i: (g.predict_proba(P["X"][i])[:, 1] > tg).mean(), "atc_sel": lambda P, i: (-P["X"][i][:, j] > tsel).mean(),
           "atc_nllsum": lambda P, i: (-P["X"][i][:, 13] > tnll).mean(), "atc_len": lambda P, i: (-P["L"][i] > tlen).mean(),
           "raw_sea": lambda P, i: -P["X"][i][:, 10].mean(), "lab30": lambda P, i: P["y"][i[:30]].mean(), "lab100": lambda P, i: P["y"][i[:100]].mean(),
           "true": lambda P, i: P["y"][i].mean()}
    allidx = np.arange(base["n"])
    level = {k: f(base, allidx) for k, f in est.items() if not k.startswith("lab")}
    rows = []
    for cond in ("full_q4", "full_q8", "full_paraphrase"):
        post = pool(f"{DATA}/regression/{cond}.jsonl.gz")
        print(f"{cond}: accuracy {base['y'].mean():.3f} -> {post['y'].mean():.3f}", flush=True)
        for rep in range(REPS):
            r = np.random.default_rng(rep)
            ser = {k: [] for k in est}
            for t in range(PRE + POST):
                P = base if t < PRE else post
                i = r.choice(P["n"], N, replace=True)
                for k, f in est.items():
                    ser[k].append(f(P, i))
            for k in est:
                if k == "true":
                    continue
                x = np.array(ser[k])
                cal, oos, after = x[:CAL], x[CAL:PRE], x[PRE:]
                mu, sd = cal.mean(), max(cal.std(), 1e-9)
                cz = mu - 2.576 * sd
                rows.append(dict(cond=cond, rep=rep, est=k, level=level.get(k, base["y"].mean()), drop=oos.mean() - after.mean(),
                                 snr=(oos.mean() - after.mean()) / max(oos.std(), 1e-9), fa=np.mean([(oos[:10] < cz).any(), (oos[10:] < cz).any()]),
                                 power=(after < cz).any(), power_up=(after > mu + 2.576 * sd).any(),
                                 true_drop=np.mean(ser["true"][:PRE]) - np.mean(ser["true"][PRE:])))
    D = pd.DataFrame(rows)
    D.to_csv(f"{RESULTS}/regression_streams.csv", index=False)
    return D


def regression_tables():
    D = regression_streams()
    A = D.groupby(["cond", "est"]).agg(level=("level", "mean"), drop=("drop", "mean"), snr=("snr", "median"), power=("power", "mean"),
                                       up=("power_up", "mean"), fa=("fa", "mean"), true_drop=("true_drop", "mean"))
    names = [("ES probe", "probe"), ("\\atcbest{} (Q25)", "atc_sel"), ("$\\text{ATC}_{\\text{NLL}_{\\text{sum}}}$", "atc_nllsum"),
             ("$\\text{ATC}_{\\text{length}}$", "atc_len"), ("Mean SEA (raw)", "raw_sea"), ("Judged 30", "lab30"), ("Judged 100", "lab100")]
    q = A.loc["full_q4"]
    rows = []
    for name, k in names:
        lvl = "-- & --" if k == "raw_sea" else f"{q.loc[k, 'level']:.3f} & {q.loc[k, 'drop']:.3f}"
        rows.append(f"{name} & {lvl} & {q.loc[k, 'snr']:.2f} & {q.loc[k, 'power']:.2f} & {q.loc[k, 'fa']:.2f} \\\\" + ("\n\\midrule" if k == "raw_sea" else ""))
    write_table("tab_quant", rows)
    rows = []
    for name, k in names:
        cells = []
        for cond in ("full_paraphrase", "full_q8"):
            a = A.loc[(cond, k)]
            cells.append(("--" if k == "raw_sea" else f"{a['drop']:.3f}".replace("-0.000", "0.000")) + f" & {a['snr']:.2f} & {a['power']:.2f} & {a['up']:.2f}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\" + ("\n\\midrule" if k == "raw_sea" else ""))
    write_table("tab_regctrl", rows)
    print("true stream drops:", A.groupby("cond").true_drop.first().round(3).to_dict())


def qa_table():
    R = [r for r in read_jsonl(f"{DATA}/qa/qa_qwen3-4b.jsonl.gz") if len(r["entropy_profile"]) >= 2]
    d = {}
    for b in QA_BENCH:
        rs = [r for r in R if r["bench"] == b]
        d[b] = dict(X=np.stack([feats(r["entropy_profile"], r["selected_logprobs"]) for r in rs]), y=np.array([qa_correct(r) for r in rs]),
                    L=np.mean([len(r["entropy_profile"]) for r in rs]))
    rng = np.random.default_rng(42)
    transfer = {}
    for name, benches in (("all", BENCH), ("mid", ("mathhendrycks", "scibench"))):
        estimate, _ = train_estimators(*qwen_stem_estimators(benches, rng))
        transfer[name] = {b: estimate(d[b]["X"]) for b in QA_BENCH}
    rows = []
    for b in QA_BENCH:
        a = stat_aurocs(d[b]["X"], d[b]["y"])
        best = max(STATS, key=a.get)
        t = transfer["all"][b]
        rows.append(f"{QA_NAMES[b]} & {d[b]['y'].mean():.3f} & {d[b]['L']:.0f} & {a[best]:.2f} ({best}) & {t['probe']:.2f} / {t['sel']:.2f} \\\\")
    D = within_sweep(holdout(d), (1, 2))
    rows += ["\\midrule", "\\textbf{Within QA} & \\textbf{Probe} & \\textbf{\\atcbest{}} & \\textbf{$\\text{ATC}_{\\text{SEA}}$} & \\textbf{Probe, avg.} \\\\", "\\midrule"]
    for k, h in D.groupby("k"):
        m = h.median(numeric_only=True)
        rows.append(f"$k{{=}}{k}$ ({len(h)} pairs) & " + " & ".join(f"{m[f'e_{c}']:.3f}" for c in ("probe", "sel", "sea", "avg")) + " \\\\")
    write_table("tab_qa", rows)


def topk_table():
    cache = f"{os.path.dirname(__file__)}/cache"
    deltas = []
    for b in ("mathhendrycks", "gsm", "olympiadbench"):
        full = torch.load(f"{cache}/phi3-3b-{b}-test.pt", map_location="cpu", weights_only=False)
        top = [it for f in sorted(glob.glob(f"{RUNS_2048}/phi3-3b-{b}-test/*.pt")) for it in torch.load(f, map_location="cpu", weights_only=False)]
        fail = np.array(full["labels"])
        assert len(top) == len(fail) and all(int(not it["success"]) == l for it, l in zip(top, fail))
        stats_full = profile_stats(full["profiles"], fail)
        stats_top = profile_stats([it["entropy_profile"].float().numpy() for it in top], fail)
        deltas.append(stats_full - stats_top)
    rows = []
    for i, n in enumerate(TOPK_STATS):
        rows.append(f"{n} & " + " & ".join(f"${v[i]:+.3f}$" for v in deltas) + " \\\\")
    write_table("tab_topk_ablation", rows)


def profile_stats(profiles, fail):
    M = np.array([[p.mean(), p.std(), p.max(), *np.percentile(p, [10, 25, 50, 75, 90]), skew(p), kurtosis(p), p.sum()] for p in (np.asarray(x, np.float32) for x in profiles)])
    a = np.array([roc_auc_score(fail, M[:, j]) for j in range(11)])
    a[[8, 9]] = 1 - a[[8, 9]]
    return a


if __name__ == "__main__":
    if os.path.isdir(f"{os.path.dirname(__file__)}/cache"):
        topk_table()
    api_tables()
    regression_tables()
    qa_table()
