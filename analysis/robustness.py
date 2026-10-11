import glob
import itertools
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import BENCH, CAP, DATA, RESULTS, load, pooled, split, thr

MONO = [-1] * 8 + [0, 0] + [-1] * 4
TRAJ_LEN = [9, 16, 17]


def trajectory(H, lp):
    H, lp = np.asarray(H, float), np.asarray(lp, float)
    n = len(H)
    k = max(1, n // 10)
    slope = np.polyfit(np.linspace(0, 1, n), H, 1)[0] if n > 2 else 0.0
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, (H < 0.1).astype(int), 0])))[::2]
    ac1 = np.corrcoef(H[:-1], H[1:])[0, 1] if n > 2 and H.std() > 0 else 0.0
    hist = np.histogram(H, bins=np.array([0, .1, .5, 1, 2, 3, 10]))[0]
    pr = hist / hist.sum()
    nll = -lp
    return np.array([H[:k].mean(), H[-k:].mean(), H[:k].mean() - H[-k:].mean(), slope, (H > 2).sum() / n * 100, (H < 0.1).mean(),
                     (runs.max() / n) if len(runs) else 0.0, ac1, -(pr[pr > 0] * np.log(pr[pr > 0])).sum(), np.log(n), (H > 1).mean(),
                     np.median(np.abs(np.diff(H))) if n > 1 else 0.0, (nll > 1).mean(), np.quantile(nll, .9), (np.exp(lp) < 0.5).mean(),
                     H[n // 2:].mean() - H[:n // 2].mean(), np.log1p(n) * H.mean(), float(n >= CAP)])


def bank(model):
    out = {}
    for b in BENCH:
        X, y = load(model, b)
        rows, seas = [], []
        for fn in sorted(glob.glob(f"{DATA}/runs/{model}-{b}-test/*.pt")):
            for r in torch.load(fn, weights_only=False):
                rows.append(trajectory(r["entropy_profile"], r["selected_logprobs"]))
                seas.append(float(np.asarray(r["entropy_profile"], float).sum()))
        E, seas = np.array(rows), np.array(seas)
        ok = np.mean(np.abs(seas - X[:, 10]) < 1e-2 * np.maximum(1, np.abs(seas))) if len(seas) == len(X) else 0.0
        if ok < 0.98:
            order = np.argsort(seas)
            ss = seas[order]
            pos = np.searchsorted(ss, X[:, 10])
            lo, hi = np.clip(pos - 1, 0, len(ss) - 1), np.clip(pos, 0, len(ss) - 1)
            near = order[np.where(np.abs(ss[lo] - X[:, 10]) <= np.abs(ss[hi] - X[:, 10]), lo, hi)]
            good = np.abs(seas[near] - X[:, 10]) <= 1e-4 * np.maximum(1.0, np.abs(X[:, 10]))
            E = E[near].astype(float)
            E[~good] = np.nan
        out[f"{b}_ext"] = E
    np.savez(f"{RESULTS}/bank_{model}.npz", **out)


def gbm(**kw):
    return HistGradientBoostingClassifier(**{**dict(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, class_weight="balanced", random_state=42), **kw})


def search(config, model):
    z = np.load(f"{RESULTS}/bank_{model}.npz")
    d = split(model)
    for b in BENCH:
        d[b]["ext"] = np.nan_to_num(z[f"{b}_ext"], nan=0.0, posinf=0.0, neginf=0.0)
    ext = config in ("ext", "extnolen", "gbm_deep_ext", "bag5_ext")
    keep = [i for i in range(18) if i not in TRAJ_LEN] if config == "extnolen" else list(range(18))
    feat = lambda b: np.hstack([d[b]["X"], d[b]["ext"][:, keep]]) if ext else d[b]["X"]

    def fit(X, y):
        if config in ("gbm_deep", "gbm_deep_ext"):
            return [gbm(max_depth=6, max_iter=400, learning_rate=0.03).fit(X, y)]
        if config in ("bag5", "bag5_ext"):
            r = np.random.default_rng(0)
            return [gbm(random_state=s).fit(X[i], y[i]) for s in range(5) for i in [r.choice(len(y), len(y), replace=True)]]
        if config == "et":
            return [ExtraTreesClassifier(400, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=1).fit(X, y)]
        if config == "rf_deep":
            return [RandomForestClassifier(400, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=1).fit(X, y)]
        if config == "lr_poly":
            return [make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                  LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")).fit(X, y)]
        if config == "mlp":
            return [make_pipeline(StandardScaler(), MLPClassifier((64, 32), early_stopping=True, max_iter=300, random_state=42, alpha=1e-3)).fit(X, y)]
        if config == "monotone":
            return [gbm(monotonic_cst=MONO).fit(X, y)]
        return [gbm().fit(X, y)]

    score = lambda ms, X: np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)
    rows = []
    for k in (2, 3):
        for G in itertools.combinations(BENCH, k):
            Xtr = np.vstack([feat(b)[d[b]["tr"]] for b in G])
            Xca = np.vstack([feat(b)[d[b]["ca"]] for b in G])
            _, ytr = pooled(d, G, "tr")
            Bca, yca = pooled(d, G, "ca")
            if len(set(ytr)) < 2 or len(set(yca)) < 2:
                continue
            if config == "votes":
                taus = np.array([thr(-Bca[:, j], yca) for j in range(14)])
                vote = lambda B: (-B > taus).astype(float)
                Xtr, Xca = np.hstack([Xtr, vote(pooled(d, G, "tr")[0])]), np.hstack([Xca, vote(Bca)])
            ms = fit(Xtr, ytr)
            s_ca = score(ms, Xca)
            if config == "ens_sel":
                jsel = int(np.argmax([roc_auc_score(yca, -Bca[:, j]) for j in range(14)]))
                fuse = lambda s, B: 0.5 * rankdata(s) / len(s) + 0.5 * rankdata(-B[:, jsel]) / len(s)
            tau = thr(fuse(s_ca, Bca) if config == "ens_sel" else s_ca, yca)
            if config == "iso":
                cal = IsotonicRegression(out_of_bounds="clip").fit(s_ca, yca)
            if config == "platt":
                logit = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
                cal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s_ca)[:, None], yca)
            for b in [b for b in BENCH if b not in G]:
                X, B, y = feat(b), d[b]["X"], d[b]["y"]
                if config == "votes":
                    X = np.hstack([X, vote(B)])
                s = score(ms, X)
                if config == "ens_sel":
                    f = fuse(np.concatenate([s_ca, s]), np.vstack([Bca, B]))
                    est = (f[len(s_ca):] > thr(f[:len(s_ca)], yca)).mean()
                elif config == "iso":
                    est = cal.predict(s).mean()
                elif config == "platt":
                    est = cal.predict_proba(logit(s)[:, None])[:, 1].mean()
                else:
                    est = (s > tau).mean()
                r = dict(model=model, k=k, group="+".join(G), held=b, true=y.mean(), est=est)
                if config == "agg":
                    r["mix"] = 0.5 * est + 0.5 * s.mean()
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{RESULTS}/search_{config}_{model}.csv", index=False)


def seeds(model, seed):
    seed = int(seed)
    d = split(model, np.random.default_rng(seed))
    rows = []
    for k in (1, 2, 3, 4):
        for G in itertools.combinations(BENCH, k):
            Xtr, ytr = pooled(d, G, "tr")
            Xca, yca = pooled(d, G, "ca")
            if ytr.min() == ytr.max() or yca.min() == yca.max():
                continue
            g = gbm(random_state=seed).fit(Xtr, ytr)
            tg = thr(g.predict_proba(Xca)[:, 1], yca)
            j = int(np.argmax([roc_auc_score(yca, -Xca[:, i]) for i in range(14)]))
            tsel = thr(-Xca[:, j], yca)
            for b in [b for b in BENCH if b not in G]:
                X, y = d[b]["X"], d[b]["y"]
                rows.append(dict(model=model, k=k, group="+".join(G), held=b, true=y.mean(), probe=(g.predict_proba(X)[:, 1] > tg).mean(), sel=(-X[:, j] > tsel).mean()))
    pd.DataFrame(rows).to_csv(f"{RESULTS}/seeds_{model}_{seed}.csv", index=False)


def truncation(model):
    d = split(model)
    rows = []
    for k in (2, 3):
        for G in itertools.combinations(BENCH, k):
            keep = lambda b, part: d[b][part][d[b]["L"][d[b][part]] < CAP]
            Xtr, ytr = np.vstack([d[b]["X"][keep(b, "tr")] for b in G]), np.concatenate([d[b]["y"][keep(b, "tr")] for b in G])
            Xca, yca = np.vstack([d[b]["X"][keep(b, "ca")] for b in G]), np.concatenate([d[b]["y"][keep(b, "ca")] for b in G])
            if len(set(ytr)) < 2 or len(set(yca)) < 2:
                continue
            g = gbm().fit(Xtr, ytr)
            tg = thr(g.predict_proba(Xca)[:, 1], yca)
            j = int(np.argmax([roc_auc_score(yca, -Xca[:, i]) for i in range(14)]))
            tsel = thr(-Xca[:, j], yca)
            for b in [b for b in BENCH if b not in G]:
                m = d[b]["L"] < CAP
                if m.sum() < 20 or len(set(d[b]["y"][m])) < 2:
                    continue
                X, y = d[b]["X"][m], d[b]["y"][m]
                rows.append(dict(model=model, k=k, group="+".join(G), held=b, true=y.mean(), probe=(g.predict_proba(X)[:, 1] > tg).mean(), sel=(-X[:, j] > tsel).mean()))
    pd.DataFrame(rows).to_csv(f"{RESULTS}/truncation_{model}.csv", index=False)


if __name__ == "__main__":
    {"bank": bank, "search": search, "seeds": seeds, "truncation": truncation}[sys.argv[1]](*sys.argv[2:])
