import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import BENCH, RESULTS, STATS, gbm, pooled, split, thr

F3 = [2, 10, 13]


def labeled(y, ind, draw, ns):
    out = {}
    for n in ns:
        lab, ppi = [], []
        for _ in range(20):
            idx = draw.choice(len(y), min(n, len(y)), replace=False)
            lab.append(y[idx].mean())
            ppi.append(ind.mean() + (y[idx] - ind[idx]).mean())
        out[f"lab{n}"] = np.mean(np.abs(np.array(lab) - y.mean()))
        out[f"ppi{n}"] = np.mean(np.abs(np.array(ppi) - y.mean()))
    return out


def tuned(y, ind, draw):
    out = {}
    for n in (30, 100):
        lab, plain, tuned_ = [], [], []
        for _ in range(20):
            idx = draw.choice(len(y), min(n, len(y)), replace=False)
            yl, il = y[idx], ind[idx]
            lam = float(np.clip(np.cov(yl, il)[0, 1] / il.var(), 0, 1)) if il.var() > 0 else 0.0
            lab.append(abs(yl.mean() - y.mean()))
            plain.append(abs(yl.mean() + (ind.mean() - il.mean()) - y.mean()))
            tuned_.append(abs(yl.mean() + lam * (ind.mean() - il.mean()) - y.mean()))
        out[f"full_lab{n}"], out[f"ppi{n}_plain"], out[f"ppi{n}_tuned"] = np.mean(lab), np.mean(plain), np.mean(tuned_)
    return out


def sweep(model):
    d = split(model)
    draw, draw_full = np.random.default_rng(0), np.random.default_rng(0)
    rows = []
    for k in (1, 2, 3, 4):
        for G in itertools.combinations(BENCH, k):
            Xtr, ytr = pooled(d, G, "tr")
            Xca, yca = pooled(d, G, "ca")
            if ytr.min() == ytr.max():
                continue
            g14, g10, g3 = gbm().fit(Xtr, ytr), gbm().fit(Xtr[:, :10], ytr), gbm().fit(Xtr[:, F3], ytr)
            rf = RandomForestClassifier(100, max_depth=5, class_weight="balanced_subsample", random_state=42, n_jobs=1).fit(Xtr, ytr)
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
            probes = {"GBM": lambda X: g14.predict_proba(X)[:, 1], "GBM10": lambda X: g10.predict_proba(X[:, :10])[:, 1],
                      "GBM3": lambda X: g3.predict_proba(X[:, F3])[:, 1], "RF": lambda X: rf.predict_proba(X)[:, 1],
                      "LR": lambda X: lr.predict_proba(sc.transform(X))[:, 1]}
            T = {n: thr(f(Xca), yca) for n, f in probes.items()}
            Tstat = [thr(-Xca[:, j], yca) for j in range(14)] if yca.min() != yca.max() else None
            for b in [b for b in BENCH if b not in G]:
                X, y = d[b]["X"], d[b]["y"]
                s = {n: f(X) for n, f in probes.items()}
                r = dict(model=model, k=k, group="+".join(G), held=b, true=y.mean())
                r.update({f"{n} thr": (s[n] > T[n]).mean() for n in probes})
                r.update({f"{n} avg": s[n].mean() for n in probes})
                if Tstat is not None:
                    r.update({f"atc {n}": (-X[:, j] > Tstat[j]).mean() for j, n in enumerate(STATS)})
                ind = (s["GBM"] > T["GBM"]).astype(float)
                if k in (2, 3):
                    r.update(labeled(y, ind, draw, (10, 30, 100)))
                if k == 3:
                    r.update(tuned(y, ind, draw_full))
                rows.append(r)
        print(model, "k", k, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{RESULTS}/stem_sweep_{model}.csv", index=False)


if __name__ == "__main__":
    sweep(sys.argv[1])
