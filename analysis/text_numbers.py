import glob
import itertools
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, pearsonr
from sklearn.metrics import roc_auc_score

from appendix_results import QA_BENCH, gsm_correct, holdout, qa_correct, qwen_stem_estimators, train_estimators, within_sweep
from common import BENCH, DATA, MODELS, OUT, RESULTS, STATS, feats, length, load, pooled, read_jsonl, results, split, thr
from section_5_results import feats as chat_feats
from section_5_results import loco, rf

P = "MAE|GBM thr"
FAMILY = {"gsm", "svamp", "gsmsymbolic"}
SEARCH = ["base", "monotone", "gbm_deep", "bag5", "rf_deep", "et", "lr_poly", "mlp", "votes", "ens_sel", "ext", "extnolen", "agg", "iso", "platt"]


def r(x, n=4):
    return round(float(x), n)


def section_3():
    S = pd.read_csv(f"{RESULTS}/sensitivity_auroc.csv")
    out = {"sensitivity_range": {c: [r(S[c].min(), 3), r(S[c].max(), 3)] for c in ("Q10", "Q25", "SEA", r"NLL$_{\text{sum}}$")}}
    best = {}
    for m in ("ministral3-3b", "qwen3-8b", "qwen3-4b", "llama3-8b", "gemma3-4b", "gemma3-12b"):
        for b in ("mathhendrycks", "gsm"):
            X, y = load(m, b)
            a = [roc_auc_score(~y, -X[:, j] if j in (8, 9) else X[:, j]) for j in range(14)]
            best[f"{m}/{b}"] = [STATS[i] for i in np.argsort(a)[::-1][:3]]
    out["appendix_top3_statistics"] = best
    return out


def bootstrap_ci(sub):
    rng = np.random.default_rng(0)
    d = sub.sel - sub[P]
    meds = [np.median(np.concatenate([d[sub.model == m].values for m in rng.choice(MODELS, 9, replace=True)])) for _ in range(2000)]
    return r(d.median()), [r(np.percentile(meds, 2.5)), r(np.percentile(meds, 97.5))]


def selected_per_pair():
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
                rows += [dict(model=m, k=k, group="+".join(G), held=b, e_sel=abs(((-sign * d[b]["X"])[:, j] > tau).mean() - d[b]["y"].mean()))
                         for b in BENCH if b not in G]
    return pd.DataFrame(rows)


def section_4():
    G = pd.read_csv(f"{RESULTS}/per_group.csv")
    S = results("stem_sweep_*.csv")
    sub, k3 = G[G.k >= 3], G[G.k == 3]
    pm = k3.groupby("model")[[P, "sel", "r|GBM thr"]].median()
    out = {"probe_mae_by_k": {k: r(G[G.k == k][P].median()) for k in (1, 2, 3, 4)},
           "k3plus": {"probe": r(sub[P].median()), "atc_best": r(sub.sel.median()), "gain": r(1 - sub[P].median() / sub.sel.median()),
                      "win": r((sub.sel > sub[P]).mean()), "oracle": r(sub.oracle.median()), "atc_max": r(sub.atc_max.median())},
           "k3_per_llm": {"probe_beats_atc_best": int((pm[P] < pm.sel).sum()), "llama_gap": r(pm.loc["llama3-8b", P] - pm.loc["llama3-8b", "sel"]),
                          "mean_gain": r((1 - pm[P] / pm.sel).mean()), "r": pm["r|GBM thr"].round(4).to_dict()}}
    out["bootstrap_gain_median"], out["bootstrap_gain_ci"] = bootstrap_ci(sub)
    out["group_spread_iqr_by_k"] = G.groupby(["model", "k"])[P].agg(lambda x: x.quantile(.75) - x.quantile(.25)).groupby("k").median().round(4).to_dict()
    out["averaging_minus_threshold_by_k"] = {k: r(G[G.k == k]["MAE|GBM avg"].median() - G[G.k == k][P].median()) for k in (1, 2, 3, 4)}
    K = G[G.k >= 2].assign(bin=lambda x: pd.cut(x.wacc, np.linspace(0, 1, 11)).astype(str))
    b = K.groupby("bin").agg(thr=(P, "median"), avg=("MAE|GBM avg", "median"), n=(P, "size"))
    out["mae_by_weighted_accuracy"] = b[b.n >= 10].round(3).to_dict("index")
    out["ablation_k3plus"] = {c: r(sub[f"MAE|{c} thr"].median()) for c in ("GBM", "GBM10", "GBM3", "RF", "LR")}
    h = S[S.k == 3].assign(e=lambda x: (x["GBM thr"] - x.true).abs()).groupby(["model", "group"]).mean(numeric_only=True)
    out["label_crossover_n"] = r((np.median([h[f"lab{n}"].median() * np.sqrt(n) for n in (10, 30, 100)]) / h.e.median()) ** 2, 1)
    out["a_priori_mae_per_llm"] = {name: S[S.group == grp].assign(e=lambda q: (q["GBM thr"] - q.true).abs()).groupby("model").e.mean().round(3).to_dict()
                                   for name, grp in (("extremes", "gsm+olympiadbench"), ("midrange", "mathhendrycks+scibench"))}
    A = S[["model", "k", "group", "held", "true", "GBM thr"]].merge(selected_per_pair(), on=["model", "k", "group", "held"])
    A["e_probe"] = (A["GBM thr"] - A.true).abs()
    in_group = A.group.apply(lambda g: bool(FAMILY & set(g.split("+"))))
    for name, m in (("sibling_excluded", ~(in_group & A.held.isin(FAMILY))), ("family_removed", ~in_group & ~A.held.isin(FAMILY))):
        g = A[m & (A.k >= 3)].groupby(["model", "k", "group"])[["e_probe", "e_sel"]].mean()
        out[name] = {"probe": r(g.e_probe.median()), "atc_best": r(g.e_sel.median())}
    D = pd.read_csv(f"{RESULTS}/decoupling.csv")
    out["decoupling"] = {g: {"r2_threshold": r(pearsonr(x.auroc, x.mae)[0] ** 2, 3), "r2_averaging": r(pearsonr(x.auroc, x.mae_avg)[0] ** 2, 3)} for g, x in D.groupby("group")}
    out["decoupling"].update({"auroc_range": [r(D.auroc.min(), 3), r(D.auroc.max(), 3)], "probe_auroc_median": r(D.auroc.median(), 3),
                              "sea_auroc_median": r(D.auroc_sea.median(), 3)})
    out["length_atc_k3"] = r(k3.atc_len.median())
    ser = pd.read_json(f"{RESULTS}/change_series.json")
    rows = []
    for _, s in ser.iterrows():
        p, l = np.array(s.probe), np.array(s.lab100)
        trig = p < p[:20].mean() - 2.576 * max(p[:20].std(), 1e-9)
        conf = trig & (l < l[:20].mean() - 2.576 * max(l[:20].std(), 1e-9))
        rows.append(dict(setting=s.setting, judged_per_window=trig[20:40].mean() * 100, false_alarm=np.mean([conf[20:30].any(), conf[30:40].any()]), power=conf[40:].any()))
    out["trigger_then_judge"] = pd.DataFrame(rows).groupby("setting").mean().round(3).to_dict("index")
    return out


def robustness():
    out = {}
    for s in (1, 2):
        x = results(f"seeds_*_{s}.csv")
        g = x.assign(ep=(x.probe - x.true).abs(), es=(x.sel - x.true).abs()).groupby(["model", "k", "group"])[["ep", "es"]].mean().reset_index()
        g = g[g.k >= 3]
        out[f"seed_{s}_k3plus"] = {"probe": r(g.ep.median()), "atc_best": r(g.es.median()), "gain": r(1 - g.ep.median() / g.es.median())}
    x = results("truncation_*.csv")
    x = x[x.k == 3]
    g = x.assign(ep=(x.probe - x.true).abs(), es=(x.sel - x.true).abs()).groupby(["model", "group"])[["ep", "es"]].mean()
    out["truncated_removed_k3"] = {"probe": r(g.ep.median()), "atc_best": r(g.es.median()), "gain": r(1 - g.ep.median() / g.es.median())}
    search = {}
    for c in SEARCH:
        x = results(f"search_{c}_*.csv")
        for col in ("est", "mix") if c == "agg" else ("est",):
            g = x.assign(e=(x[col] - x.true).abs()).groupby(["model", "k", "group"]).e.mean().reset_index()
            search[c if col == "est" else "agg_mix"] = r(g[g.k == 3].e.median())
    out["alternative_probes_k3"] = search
    return out


def section_5():
    import section_5_results as s5
    X, y, cat, L = [], [], [], []
    for f in sorted(glob.glob(f"{DATA}/runs/phi3-3b-mtbench-test/*.pt")):
        for conv in torch.load(f, weights_only=False):
            for m in conv["messages"]:
                try:
                    s = float(m.get("judge_score"))
                except (TypeError, ValueError):
                    continue
                H, lp = m.get("entropy_profile"), m.get("selected_logprobs")
                if m.get("role") != "assistant" or H is None or len(H) < 2 or lp is None or len(lp) < 1:
                    continue
                X.append(chat_feats(H, lp))
                y.append(s / 10)
                cat.append(conv["primary_category"])
    X, y, cat = np.array(X), np.array(y), np.array(cat)
    est, tru, _ = loco(X, y, cat, rf)
    W = pd.read_csv(f"{RESULTS}/within_category.csv")
    w = W[W.category.isin(["Coding & Debugging", "Math"])]
    return {"wildbench_turns_categories": {m: [len(s5.load_wildbench(m)[1]), len(set(s5.load_wildbench(m)[2]))] for m in ("phi3-3b", "oss-20b")},
            "mtbench_phi": {"n": len(y), "categories": len(set(cat)), "probe_r": r(pearsonr(est, tru)[0], 3),
                            "mean_entropy_r": r(pearsonr(pd.Series(X[:, 0]).groupby(cat).mean(), pd.Series(y).groupby(cat).mean())[0], 3)},
            "within_coding_math_probe": {f"{a}/{b}": r(v, 3) for a, b, v in zip(w.model, w.category, w.probe)}}


def appendix():
    lab = json.load(open(f"{DATA}/api/labels_gpt41mini16k.json"))
    sv = json.load(open(f"{DATA}/api/labels_gpt41mini16k_selfval.json"))
    ks = [k for k in lab if k in sv]
    ol = [len(x["entropy_profile"]) for x in read_jsonl(f"{DATA}/api/gen_gpt41mini16k_olympiadbench.jsonl.gz") if x["custom_id"] in lab]
    A = pd.read_csv(f"{RESULTS}/api_sweep.csv")
    out = {"api": {"validator_agreement": r(np.mean([lab[k] == sv[k] for k in ks]), 3), "olympiad_mean_length": r(np.mean(ol), 0),
                   "olympiad_p95_length": r(np.percentile(ol, 95), 0), "probe_beats_constant_by_k": A.assign(w=A.e_probe < A.e_const).groupby("k").w.mean().round(2).to_dict()}}
    pools = {c: read_jsonl(f"{DATA}/regression/{c}.jsonl.gz") for c in ("full_base", "full_q4", "full_q8", "full_paraphrase")}
    base = {x["idx"]: gsm_correct(x) for x in pools["full_base"]}
    reg = {}
    for c in ("full_q4", "full_q8", "full_paraphrase"):
        cond = {x["idx"]: gsm_correct(x) for x in pools[c]}
        ks = [i for i in base if i in cond]
        lost, gained = sum(base[i] and not cond[i] for i in ks), sum(cond[i] and not base[i] for i in ks)
        reg[c] = {"accuracy": r(np.mean(list(cond.values())), 3), "lost": lost, "gained": gained, "mcnemar_p": r(binomtest(lost, lost + gained, .5).pvalue)}
    reg["base_accuracy"] = r(np.mean(list(base.values())), 3)
    its = [it for f in sorted(glob.glob(f"{DATA}/runs/qwen3-4b-gsmsymbolic-test/*.pt")) for it in torch.load(f, weights_only=False) if it.get("success") is not None]
    reg["exact_match_vs_validator"] = r(np.mean([gsm_correct(dict(generation=it["generation"], reference=it["reference"])) == bool(it["success"]) for it in its]), 3)
    reg["truncated_share"] = {c: r(np.mean([x["truncated"] for x in pools[c]])) for c in pools}
    reg["mean_length"] = {"calibration_math_scibench": r(np.concatenate([length(load("qwen3-4b", b)[0]) for b in ("mathhendrycks", "scibench")]).mean(), 0),
                          "bf16": r(np.mean([len(x["entropy_profile"]) for x in pools["full_base"]]), 0)}
    reg["mean_sea_correct"] = {c: r(np.mean([sum(x["entropy_profile"]) for x in pools[c] if gsm_correct(x)]), 1) for c in ("full_base", "full_q8")}
    out["regression"] = reg
    R = [x for x in read_jsonl(f"{DATA}/qa/qa_qwen3-4b.jsonl.gz") if len(x["entropy_profile"]) >= 2]
    d = {b: dict(X=np.stack([feats(x["entropy_profile"], x["selected_logprobs"]) for x in R if x["bench"] == b]), y=np.array([qa_correct(x) for x in R if x["bench"] == b]))
         for b in QA_BENCH}
    rng = np.random.default_rng(42)
    transfer = {}
    for name, benches in (("all_stem", BENCH), ("midrange", ("mathhendrycks", "scibench"))):
        estimate, _ = train_estimators(*qwen_stem_estimators(benches, rng))
        transfer[name] = {b: {k: r(v, 3) for k, v in estimate(d[b]["X"]).items()} for b in QA_BENCH}
    D = within_sweep(holdout(d), (1, 2))
    out["qa"] = {"transfer": transfer, "within_median_error": D.groupby("k")[["e_probe", "e_sel", "e_sea", "e_nllsum", "e_avg"]].median().round(3).to_dict("index"),
                 "stem_mean_length_qwen3_4b": r(np.concatenate([length(load("qwen3-4b", b)[0]) for b in BENCH]).mean(), 0)}
    return out


if __name__ == "__main__":
    numbers = {"section_3": section_3(), "section_4": section_4(), "robustness": robustness(), "section_5": section_5(), "appendix": appendix()}
    with open(f"{OUT}/numbers.json", "w") as f:
        json.dump(numbers, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(numbers, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
