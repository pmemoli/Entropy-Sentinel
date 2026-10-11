import glob

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from common import DATA, MODELS, RESULTS, TEX_MODEL, write_table

FEATURES = ["mean", "std", "max", "q10", "q25", "q50", "q75", "q90", "skew.", "kurt.", "se\\_sum", "nll\\_avg", "nll\\_max", "nll\\_sum", "lntp", "mtp", "ppl"]


def feats(H, lp):
    H, lp = np.asarray(H, float), np.asarray(lp, float)
    nll = -lp
    return np.array([H.mean(), H.std(ddof=1) if len(H) > 1 else 0, H.max(), *np.quantile(H, [.1, .25, .5, .75, .9]),
                     stats.skew(H), stats.kurtosis(H, fisher=False), H.sum(), nll.mean(), nll.max(), nll.sum(),
                     np.exp(lp.mean()), np.exp(lp.min()), np.exp(nll.mean())])


def load_wildbench(model):
    X, y, cat, L = [], [], [], []
    for f in sorted(glob.glob(f"{DATA}/runs/{model}-wildbench-test/*.pt")):
        for conv in torch.load(f, weights_only=False):
            for m in conv["messages"]:
                if m.get("role") != "assistant":
                    continue
                H, lp = m.get("entropy_profile"), m.get("selected_logprobs")
                try:
                    s = float(m.get("judge_score"))
                except (TypeError, ValueError):
                    continue
                if H is None or len(H) < 2 or lp is None or len(lp) < 1:
                    continue
                X.append(feats(H, lp))
                y.append(s / 10)
                cat.append(conv["primary_category"])
                L.append(len(H))
    return np.array(X), np.array(y), np.array(cat), np.array(L, float)


def rf(Xtr, ytr, ctr, Xte):
    w = pd.Series(ctr).map(1 / pd.Series(ctr).value_counts()).values
    m = RandomForestRegressor(200, max_depth=8, random_state=42, n_jobs=-1).fit(Xtr, ytr, sample_weight=w / w.mean())
    return m.predict(Xte), m.predict(Xtr)


def lin(Xtr, ytr, ctr, Xte):
    s = StandardScaler().fit(Xtr)
    m = LinearRegression().fit(s.transform(Xtr), ytr)
    return m.predict(s.transform(Xte)), m.predict(s.transform(Xtr))


def loco(X, y, cat, fit):
    est, tru, p = {}, {}, np.zeros(len(y))
    for c in sorted(set(cat)):
        tr, te = cat != c, cat == c
        p_te, p_tr = fit(X[tr], y[tr], cat[tr], X[te])
        est[c], tru[c] = p_te.mean() - p_tr.mean(), y[te].mean() - y[tr].mean()
        p[te] = p_te - p_tr.mean()
    return pd.Series(est), pd.Series(tru), p


def boot_ci(est, tru, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(B):
        idx = rng.choice(len(est), len(est), replace=True)
        e, t = est.values[idx], tru.values[idx]
        if e.std() > 0 and t.std() > 0:
            rs.append(pearsonr(e, t)[0])
    return np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def partial_spearman(a, b, z):
    def resid(u, v):
        v1 = np.c_[np.ones(len(v)), v]
        return u - v1 @ np.linalg.lstsq(v1, u, rcond=None)[0]
    rz = rankdata(z)
    return pearsonr(resid(rankdata(a), rz), resid(rankdata(b), rz))[0]


def num(v):
    return f"${v:.2f}$" if v < 0 else f"{v:.2f}"


def ci(v):
    return f"{v:.2f}".replace("0.", ".")


def loco_and_within():
    loco_rows, within_rows = [], []
    for m in MODELS:
        X, y, cat, L = load_wildbench(m)
        est, tru, p = loco(X, y, cat, rf)
        lo, hi = boot_ci(est, tru)
        raw = pearsonr(pd.Series(X[:, 0]).groupby(cat).mean(), pd.Series(y).groupby(cat).mean())[0]
        r_len = pearsonr(*loco(L[:, None], y, cat, lin)[:2])[0]
        r_pl = pearsonr(*loco(np.c_[X, L], y, cat, rf)[:2])[0]
        loco_rows.append(dict(model=m, r=pearsonr(est, tru)[0], lo=lo, hi=hi, raw=raw, length=r_len, probe_length=r_pl))
        for c in sorted(set(cat)):
            k = cat == c
            if k.sum() >= 100:
                within_rows.append(dict(model=m, category=c, n=int(k.sum()), probe=spearmanr(p[k], y[k])[0], meanH=spearmanr(X[k, 0], y[k])[0],
                                        SEA=spearmanr(X[k, 10], y[k])[0], length=spearmanr(L[k], y[k])[0], probe_given_length=partial_spearman(p[k], y[k], L[k])))
        print(m, "done", flush=True)
    T, W = pd.DataFrame(loco_rows), pd.DataFrame(within_rows)
    T.to_csv(f"{RESULTS}/loco.csv", index=False)
    W.to_csv(f"{RESULTS}/within_category.csv", index=False)
    write_table("tab_loco_results", [f"{TEX_MODEL[r.model]} & {num(r.r)} & $[{ci(r.lo)},{ci(r.hi)}]$ & {num(r.raw)} & {num(r.length)} & {num(r.probe_length)} \\\\" for r in T.itertuples()])
    cols = ["probe", "meanH", "SEA", "length", "probe_given_length"]
    rows = [f"{TEX_MODEL[m]} & {len(x)} & " + " & ".join(num(v) for v in x[cols].median()) + " \\\\" for m, x in W.groupby("model", sort=False)]
    rows += ["\\midrule", f"\\textit{{All}} & {len(W)} & " + " & ".join(num(v) for v in W[cols].median()) + " \\\\"]
    write_table("tab_within_cat", rows)


def feature_agreement():
    rows = {}
    for m in MODELS:
        f = torch.load(f"{DATA}/features/{m}-wildbench-test.pt", weights_only=False)
        X = np.array([np.asarray(r["features:"]) for r in f])
        y = np.array([r["judge_score"] for r in f])
        cat = np.array([r["primary_category"] for r in f])
        cats = sorted(np.unique(cat))
        score = np.array([y[cat == c].mean() for c in cats])
        rows[m] = [pearsonr([X[cat == c, j].mean() for c in cats], score)[0] for j in range(17)]
    A = pd.DataFrame(rows, index=FEATURES).T.sort_values("mean", ascending=False)
    A.to_csv(f"{RESULTS}/feature_category_pearson.csv")
    out = []
    for m, v in A.iterrows():
        best = int(np.argmax(np.abs(v.values)))
        cells = [f"\\textbf{{{x:.3f}}}" if j == best else f"{x:.3f}" for j, x in enumerate(v.values)]
        out.append(f"{TEX_MODEL[m]} & " + " & ".join(cells) + " \\\\")
        if m == A.index[3]:
            out.append("\\midrule")
    write_table("tab_feature_monitoring_agreement", out)


if __name__ == "__main__":
    loco_and_within()
    feature_agreement()
