import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from common import (BENCH, CAP, EXTREMES, FIGURES, MIDRANGE, MODELS, RESULTS, SHORT_MODEL, STATS, TEX_MODEL,
                    gbm, length, load, miqr, pooled, results, split, thr, write_table)

PROBES = ["GBM", "GBM10", "GBM3", "RF", "LR"]
P = "MAE|GBM thr"


def per_group():
    S = results("stem_sweep_*.csv")
    methods = [c for c in S.columns if c.endswith((" thr", " avg")) or c.startswith("atc ")]
    rows = []
    for (m, k, g), h in S.groupby(["model", "k", "group"]):
        r = dict(model=m, k=k, group=g)
        for c in methods:
            r[f"MAE|{c}"] = np.abs(h[c] - h["true"]).mean()
            r[f"r|{c}"] = pearsonr(h[c], h["true"])[0] if h["true"].std() > 0 else np.nan
        rows.append(r)
    G = pd.DataFrame(rows)
    acc = {(m, b): load(m, b)[1] for m in MODELS for b in BENCH}
    G["wacc"] = [sum(acc[(m, b)].sum() for b in g.split("+")) / sum(len(acc[(m, b)]) for b in g.split("+")) for m, g in zip(G.model, G.group)]
    G = G.merge(selected_atc(), on=["model", "k", "group"]).merge(length_controls(), on=["model", "k", "group"])
    stat_cols = [f"MAE|atc {n}" for n in STATS]
    best = {m: min(stat_cols, key=lambda c: G[G.model == m][c].median()) for m in MODELS}
    G["atc_max"] = G["MAE|atc max"]
    G["oracle"] = [r[best[r.model]] for _, r in G.iterrows()]
    G.to_csv(f"{RESULTS}/per_group.csv", index=False)
    return S, G


def selected_atc():
    sign = np.array([1] * 8 + [-1, -1] + [1] * 4)
    rows = []
    for m in MODELS:
        d = split(m)
        for k in (1, 2, 3, 4):
            for G in itertools.combinations(BENCH, k):
                Xca, yca = pooled(d, G, "ca")
                if yca.min() == yca.max():
                    continue
                S = -sign * Xca
                j = int(np.argmax([roc_auc_score(yca, S[:, i]) for i in range(14)]))
                tau = thr(S[:, j], yca)
                mae = np.mean([abs(((-sign * d[b]["X"])[:, j] > tau).mean() - d[b]["y"].mean()) for b in BENCH if b not in G])
                rows.append(dict(model=m, k=k, group="+".join(G), sel_stat=STATS[j], sel=mae))
    return pd.DataFrame(rows)


def length_controls():
    rows = []
    for m in MODELS:
        d = split(m)
        for k in (1, 2, 3, 4):
            for G in itertools.combinations(BENCH, k):
                Lca = np.concatenate([d[b]["L"][d[b]["ca"]] for b in G])
                yca = np.concatenate([d[b]["y"][d[b]["ca"]] for b in G])
                if yca.min() == yca.max():
                    continue
                tL = thr(-Lca, yca)
                acc_nc = yca[Lca < CAP].mean() if (Lca < CAP).any() else yca.mean()
                held = [b for b in BENCH if b not in G]
                rows.append(dict(model=m, k=k, group="+".join(G),
                                 atc_len=np.mean([abs((-d[b]["L"] > tL).mean() - d[b]["y"].mean()) for b in held]),
                                 cap_est=np.mean([abs(acc_nc * (d[b]["L"] < CAP).mean() - d[b]["y"].mean()) for b in held])))
    return pd.DataFrame(rows)


def by_k_row(G, name, col):
    sub = G[G.k >= 3]
    win = "--" if col == "sel" else f"{(sub['sel'] > sub[col]).mean():.2f}"
    return f"{name} & " + " & ".join(miqr(G[G.k == k][col]) for k in (1, 2, 3, 4)) + f" & {win} \\\\"


def tables(G):
    h = G[G.k == 3]
    rows = []
    for m in MODELS:
        x = h[h.model == m]
        probe, ref = miqr(x[P]), miqr(x["sel"])
        bold = x[P].median() <= x["sel"].median()
        r = f"\\miqr{{{x['r|GBM thr'].median():.2f}}}{{{x['r|GBM thr'].quantile(.75) - x['r|GBM thr'].quantile(.25):.2f}}}"
        rows.append(f"{TEX_MODEL[m]} & " + (f"\\textbf{{{probe}}} & {r} & {ref}" if bold else f"{probe} & {r} & \\textbf{{{ref}}}") + " \\\\")
    write_table("tab_apriori_results", rows)
    write_table("tab_es_vs_atc", [by_k_row(G, "Probe, thr.\\ (default)", P), by_k_row(G, "Probe, avg.", "MAE|GBM avg"),
                                  by_k_row(G, "RF probe, thr.", "MAE|RF thr"), by_k_row(G, "LR probe, thr.", "MAE|LR thr"), "\\midrule",
                                  by_k_row(G, "\\atcbest{}", "sel"), "\\midrule", "\\multicolumn{6}{l}{\\textit{Length-only controls}} \\\\",
                                  by_k_row(G, "$\\text{ATC}_{\\text{length}}$", "atc_len"), by_k_row(G, "Cap-hit share", "cap_est")])
    sub = G[G.k >= 3]
    write_table("tab_ablations", [f"{n} & {miqr(sub[f'MAE|{c} thr'])} & {miqr(sub[f'MAE|{c} avg'])} & {miqr(sub[f'r|{c} thr'])} \\\\"
                                  for n, c in [("Boosted trees", "GBM"), ("Boosted trees (10D)", "GBM10"), ("Boosted trees (3D)", "GBM3"),
                                               ("Random forest", "RF"), ("Logistic reg.", "LR")]])
    write_table("tab_hindsight", [by_k_row(G, "\\atcmax{} (best fixed statistic)", "atc_max"), by_k_row(G, "\\atcoracle{} (best statistic per LLM)", "oracle")])


def accuracy_figure(G):
    K = G[G.k >= 2].copy()
    K["bin"] = pd.cut(K.wacc, np.linspace(0, 1, 11))
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    for col, label, c, ls, band in [("MAE|GBM avg", "probe, probability averaging (Eq. 1)", "tab:gray", "--", True),
                                    ("sel", "ATC$_{\\mathrm{best}}$ (single statistic)", "tab:orange", ":", False),
                                    (P, "probe, threshold rule (Eq. 2)", "tab:blue", "-", True)]:
        s = K.groupby("bin", observed=False)[col].agg(med="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75), n="size").reset_index()
        s = s[s.n >= 10]
        x = [b.mid for b in s.bin]
        ax.plot(x, s.med, marker="o", ms=4, lw=2, color=c, ls=ls, label=label)
        if band:
            ax.fill_between(x, s.q1, s.q3, alpha=.10, color=c)
    ax.set_xlabel("training-group weighted accuracy ($k\\geq 2$)")
    ax.set_ylabel("held-out MAE (median)")
    ax.set_ylim(0.04, 0.16)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/all_llm_accuracy_v7.pdf")


def example_figure():
    names = {"gsm": "GSM8K", "svamp": "SVAMP", "gsmsymbolic": "GSM-Symb.", "mathhendrycks": "MATH", "theoremqa": "TheoremQA", "scibench": "SciBench",
             "matscibench": "MatSciBench", "olympiadbench": "Olympiad", "livemathbench": "LiveMath", "gpqa": "GPQA"}
    d = split("llama3-8b")
    Xtr, ytr = pooled(d, EXTREMES, "tr")
    Xca, yca = pooled(d, EXTREMES, "ca")
    g = gbm().fit(Xtr, ytr)
    tau = thr(g.predict_proba(Xca)[:, 1], yca)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    pts = []
    for b in BENCH:
        X, y = d[b]["X"], d[b]["y"]
        if b in EXTREMES:
            X, y = X[d[b]["ca"]], y[d[b]["ca"]]
        est = (g.predict_proba(X)[:, 1] > tau).mean()
        pts.append((b, est, y.mean()))
        ax.scatter(est, y.mean(), color="tab:orange" if b in EXTREMES else "tab:blue", s=36, zorder=3)
        ax.annotate(names[b], (est, y.mean()), fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("estimated accuracy")
    ax.set_ylabel("true accuracy")
    ood = np.array([(e, t) for b, e, t in pts if b not in EXTREMES])
    ax.set_title(f"Llama-3.1 8B: OOD MAE {np.abs(ood[:, 0] - ood[:, 1]).mean():.2f}, r = {np.corrcoef(ood.T)[0, 1]:.2f}", fontsize=9)
    ax.scatter([], [], color="tab:orange", label="training benchmarks (held-out 20%)")
    ax.scatter([], [], color="tab:blue", label="unseen benchmarks")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/fig1_example_v7.pdf")


def all_models_figures(S):
    names = {m: SHORT_MODEL[m].replace("Phi-3.5-Mini 3.8B", "Phi-3.5-Mini") for m in MODELS}
    marks = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    K2 = S[S.k == 2]

    def panel(ax, group, size, label):
        h = K2[K2.group == "+".join(group)]
        for i, m in enumerate(MODELS):
            x = h[h.model == m]
            ax.scatter(x["GBM thr"], x.true, color=cols[i], marker=marks[i], s=size, alpha=.85, zorder=3, label=names[m] if label else None, edgecolor="none")
        mae = (h["GBM thr"] - h.true).abs().groupby(h.model).mean().median()
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("estimated accuracy")
        ax.grid(alpha=.3)
        ax.set_aspect("equal")
        return mae, np.corrcoef(h["GBM thr"], h.true)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True)
    for ax, group, title in [(axes[0], EXTREMES, "Extremes (GSM8K + OlympiadBench)"), (axes[1], MIDRANGE, "Mid-range (MATH + SciBench)")]:
        mae, r = panel(ax, group, 26, ax is axes[0])
        ax.set_title(f"{title}\nmedian MAE {mae:.2f}, r = {r:.2f}", fontsize=8.5)
    axes[0].set_ylabel("true accuracy")
    fig.legend(fontsize=6.5, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{FIGURES}/fig_all_models_v7.pdf", bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    mae, r = panel(ax, MIDRANGE, 24, True)
    ax.set_ylabel("true accuracy")
    ax.set_title(f"median MAE {mae:.2f}, r = {r:.2f}", fontsize=8.5)
    ax.legend(fontsize=5.6, loc="lower right", ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/fig_all_models_mid_v7.pdf", bbox_inches="tight")


def truncation_table():
    short = {m: "\\textsc{" + SHORT_MODEL[m].replace("Phi-3.5-Mini 3.8B", "Phi-3.5-Mini") + "}" for m in MODELS}
    rows = []
    for m in MODELS:
        v = [100 * np.mean(length(load(m, b)[0]) >= CAP) for b in BENCH]
        rows.append(f"{short[m]} & " + " & ".join(f"{x:.0f}" if x >= 0.5 else ("0" if x == 0 else "<1") for x in v) + " \\\\")
    write_table("tab_truncation", rows)


def decoupling_figure(S):
    rows = []
    for m in MODELS:
        d = split(m)
        for name, group in [("Extremes", EXTREMES), ("Mid-range", MIDRANGE)]:
            Xtr, ytr = pooled(d, group, "tr")
            g = gbm().fit(Xtr, ytr)
            sw = S[(S.model == m) & (S.group == "+".join(group))].set_index("held")
            for b in [b for b in BENCH if b not in group]:
                rows.append(dict(model=m, group=name, held=b, auroc=roc_auc_score(~d[b]["y"], -g.predict_proba(d[b]["X"])[:, 1]),
                                 mae=abs(sw.loc[b, "GBM thr"] - sw.loc[b, "true"]), mae_avg=abs(sw.loc[b, "GBM avg"] - sw.loc[b, "true"]),
                                 auroc_sea=roc_auc_score(~d[b]["y"], d[b]["X"][:, 10])))
    D = pd.DataFrame(rows)
    D.to_csv(f"{RESULTS}/decoupling.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=True)
    for ax, name in zip(axes, ("Extremes", "Mid-range")):
        h = D[D.group == name]
        ax.scatter(h.auroc, h.mae, s=14, alpha=.7, color="tab:blue")
        ax.set_title(f"{name} probe: $R^2$ = {pearsonr(h.auroc, h.mae)[0] ** 2:.3f}", fontsize=9)
        ax.set_xlabel("per-instance AUROC (probe score)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("slice MAE (threshold rule)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/comparison_auroc_v7.pdf")


def labels_table(S):
    h = S[S.k == 3].assign(e_probe=lambda x: (x["GBM thr"] - x.true).abs()).groupby(["model", "group"]).mean(numeric_only=True)
    rows = [f"Probe (zero-label) & 0 & {h.e_probe.median():.3f} & -- \\\\"]
    rows += [f"Judged sample & {n} & {h[f'lab{n}'].median():.3f} & {100 * (h[f'lab{n}'] > h.e_probe).mean():.0f}\\% \\\\" for n in (10, 30, 100)]
    rows += [f"Probe + PPI & {n} & {h[f'ppi{n}_tuned'].median():.3f} & -- \\\\" for n in (30, 100)]
    write_table("tab_labels", rows)


def drift():
    train, held = list(EXTREMES), [b for b in BENCH if b not in EXTREMES]
    easy, hard = [held.index(b) for b in ("svamp", "gsmsymbolic", "mathhendrycks")], [held.index(b) for b in ("theoremqa", "livemathbench", "gpqa")]
    T, N = 30, 300
    methods = ["thr", "sel", "avg", "Judged 30", "Judged 100"]
    rows, series = [], {}
    for m in MODELS:
        rng, lrng = np.random.default_rng(7), np.random.default_rng(99)
        Xs, ys = zip(*[load(m, b) for b in train])
        Xa, ya = np.vstack(Xs), np.concatenate(ys)
        perm = rng.permutation(len(ya))
        ntr = int(0.8 * len(perm))
        Xtr, ytr, Xca, yca = Xa[perm[:ntr]], ya[perm[:ntr]], Xa[perm[ntr:]], ya[perm[ntr:]]
        g = gbm().fit(Xtr, ytr)
        tg = thr(g.predict_proba(Xca)[:, 1], yca)
        j = int(np.argmax([roc_auc_score(yca, -Xca[:, i]) for i in range(14)]))
        tsel = thr(-Xca[:, j], yca)
        pool = {b: load(m, b) for b in held}
        logits = np.zeros(len(held))
        est = {k: [] for k in ("thr", "sel", "avg")}
        true, draws = [], {30: [], 100: []}
        for t in range(T):
            base = np.zeros(len(held))
            base[easy if (t < T // 3 or t >= 2 * T // 3) else hard] += 2.0
            logits = 0.7 * logits + rng.normal(0, 0.5, len(held))
            w = np.exp(base + logits) / np.exp(base + logits).sum()
            Xw, yw = [], []
            for b, c in zip(held, rng.multinomial(N, w)):
                if c:
                    idx = rng.choice(len(pool[b][1]), c, replace=True)
                    Xw.append(pool[b][0][idx])
                    yw.append(pool[b][1][idx])
            Xw, yw = np.vstack(Xw), np.concatenate(yw)
            true.append(yw.mean())
            p = g.predict_proba(Xw)[:, 1]
            est["thr"].append((p > tg).mean())
            est["sel"].append((-Xw[:, j] > tsel).mean())
            est["avg"].append(p.mean())
            for n in (30, 100):
                draws[n].append(np.array([yw[lrng.choice(len(yw), n, replace=False)].mean() for _ in range(20)]))
        true = np.array(true)
        series[m] = (true, {k: np.array(v) for k, v in est.items()})
        for k, v in est.items():
            rows.append(dict(model=m, method=k, MAE=np.abs(np.array(v) - true).mean(), r=pearsonr(v, true)[0]))
        for n in (30, 100):
            L = np.array(draws[n])
            rows.append(dict(model=m, method=f"Judged {n}", MAE=np.mean([np.abs(L[:, i] - true).mean() for i in range(20)]),
                             r=np.mean([pearsonr(L[:, i], true)[0] for i in range(20)])))
    D = pd.DataFrame(rows)
    D.to_csv(f"{RESULTS}/drift.csv", index=False)
    out = []
    for m in MODELS:
        x = D[D.model == m].set_index("method")
        out.append(f"\\textsc{{{SHORT_MODEL[m]}}} & " + " & ".join(f"{x.loc[k, 'MAE']:.3f} & {x.loc[k, 'r']:.2f}" for k in methods) + " \\\\")
    med = D.groupby("method")[["MAE", "r"]].median()
    out += ["\\midrule", "\\textit{Median} & " + " & ".join(f"{med.loc[k, 'MAE']:.3f} & {med.loc[k, 'r']:.2f}" for k in methods) + " \\\\"]
    write_table("tab_drift", out)
    true, est = series["gemma3-12b"]
    x = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(x, true, color="black", lw=2, label="true accuracy")
    ax.plot(x, est["thr"], color="tab:blue", lw=1.6, label="ES probe (threshold rule)")
    ax.plot(x, est["sel"], color="tab:orange", lw=1.2, ls="--", label="ATC$_{\\mathrm{best}}$")
    ax.plot(x, est["avg"], color="tab:gray", lw=1.2, ls="-.", label="ES probe (averaging)")
    for v in (T // 3 + 0.5, 2 * T // 3 + 0.5):
        ax.axvline(v, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time window")
    ax.set_ylabel("accuracy")
    ax.set_title("Gemma3 12B", fontsize=9)
    ax.grid(alpha=.3)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/drift_single.pdf")


PRE, CAL, POST, N, NL = 40, 20, 10, 300, (15, 30, 50, 100)


def fit_estimators(d, group):
    Xtr, ytr = pooled(d, group, "tr")
    Xca, yca = pooled(d, group, "ca")
    g = gbm().fit(Xtr, ytr)
    j = int(np.argmax([roc_auc_score(yca, -Xca[:, i]) for i in range(14)]))
    tg, tsel = thr(g.predict_proba(Xca)[:, 1], yca), thr(-Xca[:, j], yca)
    return {"probe": lambda X: (g.predict_proba(X)[:, 1] > tg).mean(), "atc_sel": lambda X: (-X[:, j] > tsel).mean()}


def stream(est, pools, mix_pre, mix_post, rng, tilt=None):
    out = {k: [] for k in list(est) + ["true"] + [f"lab{n}" for n in NL]}
    for t in range(PRE + POST):
        mix = mix_pre if t < PRE else mix_post
        bs = list(mix)
        w = np.array([mix[b] for b in bs])
        Xw, yw = [], []
        for b, c in zip(bs, rng.multinomial(N, w / w.sum())):
            if not c:
                continue
            X, y = pools[b]
            if tilt is not None and t >= PRE:
                a = y.mean()
                target = max(a - tilt, 0.01)
                pw = np.where(y, target / max(a, 1e-9), (1 - target) / max(1 - a, 1e-9))
                idx = rng.choice(len(y), c, replace=True, p=pw / pw.sum())
            else:
                idx = rng.choice(len(y), c, replace=True)
            Xw.append(X[idx])
            yw.append(y[idx])
        Xw, yw = np.vstack(Xw), np.concatenate(yw)
        out["true"].append(yw.mean())
        for k, f in est.items():
            out[k].append(f(Xw))
        for n in NL:
            out[f"lab{n}"].append(yw[rng.choice(len(yw), n, replace=False)].mean())
    return {k: np.array(v) for k, v in out.items()}


def alarm(x):
    cal, oos, post = x[:CAL], x[CAL:PRE], x[PRE:PRE + POST]
    cz = cal.mean() - 2.576 * max(cal.std(), 1e-9)
    return dict(snr=(oos.mean() - post.mean()) / max(oos.std(), 1e-9), power=(post < cz).any(), fa=np.mean([(oos[:10] < cz).any(), (oos[10:] < cz).any()]))


def null_mix(acc, held, rng):
    base = np.mean([acc[b] for b in held])
    w = rng.dirichlet(np.ones(len(held)) * 0.7)
    a = np.array([acc[b] for b in held])
    dvec = np.where(a >= np.median(a), 1.0, -1.0)
    dvec -= dvec.mean()
    w2 = w + (base - w @ a) / (dvec @ a) * dvec
    if w2.min() < 0:
        w2 = np.clip(w2, 0, None)
        w2 /= w2.sum()
    return dict(zip(held, w2))


def drop_mix(acc, held, drop):
    base = np.mean([acc[b] for b in held])
    hard = sorted(held, key=lambda b: acc[b])[:4]
    lam = min(1.0, drop / max(base - np.mean([acc[b] for b in hard]), 1e-6))
    return {b: (1 - lam) / len(held) + (lam / len(hard) if b in hard else 0.0) for b in held}


def change_detection():
    rows, series = [], []
    for m in MODELS:
        d = split(m)
        pools = {b: (d[b]["X"], d[b]["y"]) for b in BENCH}
        est = fit_estimators(d, EXTREMES)
        held = [b for b in BENCH if b not in EXTREMES]
        acc = {b: d[b]["y"].mean() for b in held}
        flat = {b: 1.0 for b in held}
        for setting in ("step5", "null", "tilt5"):
            for rep in range(20):
                rng = np.random.default_rng(1000 + rep)
                if setting == "tilt5":
                    s = stream(est, pools, flat, flat, rng, tilt=0.05)
                else:
                    s = stream(est, pools, flat, drop_mix(acc, held, 0.05) if setting == "step5" else null_mix(acc, held, rng), rng)
                rows += [dict(model=m, setting=setting, rep=rep, estimator=k, **alarm(s[k])) for k in s if k != "true"]
                series.append(dict(model=m, setting=setting, rep=rep, probe=s["probe"].tolist(), lab100=s["lab100"].tolist()))
        est = fit_estimators(d, MIDRANGE)
        for rep in range(20):
            s = stream(est, pools, {"gsm": 1.0}, {"gsmsymbolic": 1.0}, np.random.default_rng(2000 + rep))
            rows += [dict(model=m, setting="gsm_to_symbolic", rep=rep, estimator=k, **alarm(s[k])) for k in s if k != "true"]
            series.append(dict(model=m, setting="gsm_to_symbolic", rep=rep, probe=s["probe"].tolist(), lab100=s["lab100"].tolist()))
        print(m, "change detection done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(f"{RESULTS}/change_detection.csv", index=False)
    pd.DataFrame(series).to_json(f"{RESULTS}/change_series.json")
    return D


def change_table(D):
    agg = D.groupby(["setting", "estimator"]).agg(snr=("snr", "median"), power=("power", "mean"))
    Q = pd.read_csv(f"{RESULTS}/regression_streams.csv")
    q = Q[Q.cond == "full_q4"].groupby("est").agg(snr=("snr", "median"), power=("power", "mean"))
    rows = []
    for name, k in [("ES probe", "probe"), ("\\atcbest{}", "atc_sel"), ("Judged 30", "lab30"), ("Judged 100", "lab100")]:
        a = lambda s, c: agg.loc[(s, k), c]
        rows.append(f"{name} & {a('tilt5', 'snr'):.2f} & {a('tilt5', 'power'):.2f} & {a('step5', 'snr'):.2f} & {a('step5', 'power'):.2f} & "
                    f"{a('null', 'power'):.2f} & {a('gsm_to_symbolic', 'snr'):.2f} & {a('gsm_to_symbolic', 'power'):.2f} & "
                    f"{q.loc[k, 'snr']:.2f} & {q.loc[k, 'power']:.2f} \\\\" + ("\n\\midrule" if k == "atc_sel" else ""))
    write_table("tab_change", rows)


if __name__ == "__main__":
    S, G = per_group()
    tables(G)
    accuracy_figure(G)
    example_figure()
    all_models_figures(S)
    truncation_table()
    decoupling_figure(S)
    labels_table(S)
    drift()
    change_table(change_detection())
