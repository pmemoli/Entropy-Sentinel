"""
W/AUROC sensitivity sweep analysis on MATH-test (Phi-3.5-Mini).

Reads generations from ``src/data/sensitivity/phi3-3b-math-sensitivity-t{T}-s{S}/*.pt``,
runs the Grok validator on any unlabeled items (writing labels back to the .pt
so subsequent runs skip the API), then computes per-(temperature, seed) AUROC
for each entropy-profile / log-prob summary statistic using the same metric
definitions as ``metric_predictive_power.metric_power``.

Output: ``src/analysis/sensibility_predictive_power/sensitivity_auroc.csv``
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

from src.analysis.metric_predictive_power.metric_power import (
    METRIC_NAMES,
    aurocs_from_metrics,
    per_sample_metrics,
)

RUNS_ROOT = "src/data/sensitivity"
OUT_DIR = "src/analysis/sensibility_predictive_power"
OUT_CSV = os.path.join(OUT_DIR, "sensitivity_auroc.csv")

# Original paper-run directory for the T=0.5 baseline cell.
ORIGINAL_T05_DIR = "src/data/runs/phi3-3b-mathhendrycks-test"
OG_T = 0.5
OG_SEED = 42

SUITE_RE = re.compile(r"^phi3-3b-math-sensitivity-t([0-9.]+)-s(\d+)$")


def discover_suites(root):
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        m = SUITE_RE.match(name)
        if not m:
            continue
        out.append((name, float(m.group(1)), int(m.group(2))))
    return out


def _label_one(item, evaluate_response, per_call_timeout=90):
    """Run the validator on one item with retries; returns True/False/None."""
    for i in range(6):
        try:
            with ThreadPoolExecutor(max_workers=1) as inner:
                fut = inner.submit(
                    evaluate_response,
                    item["prompt"],
                    item["generation"],
                    item["reference"],
                )
                return fut.result(timeout=per_call_timeout)
        except Exception as e:
            print(f"  validator attempt {i+1} failed: {type(e).__name__}: {e}")
            time.sleep(min(2**i, 30))
    return None


def _atomic_save(batch, path):
    """Crash-safe write: serialise to .tmp, then os.replace (atomic on POSIX)."""
    tmp = path + ".tmp"
    torch.save(batch, tmp)
    os.replace(tmp, path)


def ensure_labels(suite_dir, max_workers=8, checkpoint_every=25):
    """Validate any items missing ``success`` in parallel.

    Labels are mutated in place on items in ``batch``; the .pt file is rewritten
    atomically every ``checkpoint_every`` completions and at the end of each
    batch. KeyboardInterrupt saves what is already done before re-raising, so
    rerunning the script picks up exactly where it stopped.
    """
    from src.engine.judge_stem_scenarios import evaluate_response

    for fname in sorted(os.listdir(suite_dir)):
        if not fname.endswith(".pt"):
            continue
        full = os.path.join(suite_dir, fname)
        batch = torch.load(full, map_location="cpu", weights_only=False)
        unlabeled = [item for item in batch if "success" not in item]
        if not unlabeled:
            continue
        print(f"  {fname}: {len(unlabeled)} unlabeled items")

        done = 0
        new_since_save = 0
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = {
            pool.submit(_label_one, item, evaluate_response): item
            for item in unlabeled
        }
        try:
            for fut in as_completed(futures):
                item = futures[fut]
                result = fut.result()
                if result is not None:
                    item["success"] = result
                    new_since_save += 1
                else:
                    print(f"  WARN: gave up labeling one item in {fname}")
                done += 1
                if new_since_save >= checkpoint_every:
                    _atomic_save(batch, full)
                    new_since_save = 0
                if done % checkpoint_every == 0 or done == len(unlabeled):
                    print(f"    {done}/{len(unlabeled)} done (checkpointed)")
        except KeyboardInterrupt:
            print("\n  interrupted — saving labels collected so far and exiting...")
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            if new_since_save:
                _atomic_save(batch, full)
            raise
        finally:
            if new_since_save:
                _atomic_save(batch, full)
            pool.shutdown(wait=False)


def _aurocs_from_items(items):
    """Shared body: compute the per-stat AUROC + (n, acc) for a list of items."""
    per_sample, labels = [], []
    n_correct = 0
    for item in items:
        if "success" not in item:
            continue
        profile = item["entropy_profile"].to(torch.float32).numpy()
        lp = item["selected_logprobs"]
        if isinstance(lp, torch.Tensor):
            lp = lp.to(torch.float32).numpy()
        else:
            lp = np.asarray(lp, dtype=np.float32)
        per_sample.append(per_sample_metrics(profile, lp))
        labels.append(0 if item["success"] else 1)
        n_correct += int(bool(item["success"]))
    if not per_sample:
        return None
    if len(set(labels)) < 2:
        return {m: float("nan") for m in METRIC_NAMES}, len(per_sample), n_correct / len(per_sample)
    return aurocs_from_metrics(per_sample, labels), len(per_sample), n_correct / len(per_sample)


def _iter_dir_items(d):
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".pt"):
            continue
        batch = torch.load(
            os.path.join(d, fname),
            map_location="cpu",
            weights_only=False,
        )
        for item in batch:
            yield item


def auroc_row(suite_dir):
    return _aurocs_from_items(list(_iter_dir_items(suite_dir)))


def auroc_row_from_original(original_dir, target_prompts):
    """Replace a sensitivity-sweep cell with prompt-matched items from the
    original paper run (deterministic same-shuffle, same-seed generations
    that did not need regeneration)."""
    matched = [it for it in _iter_dir_items(original_dir) if it["prompt"] in target_prompts]
    if len(matched) < len(target_prompts):
        print(
            f"  WARN: only {len(matched)}/{len(target_prompts)} prompts "
            f"matched in {original_dir}"
        )
    return _aurocs_from_items(matched)


def collect_prompts(suite_dir):
    return {it["prompt"] for it in _iter_dir_items(suite_dir)}


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of AUROC across seeds, per temperature × metric."""
    id_vars = [c for c in ("temperature", "seed", "source", "n_samples", "accuracy") if c in df.columns]
    long = df.melt(
        id_vars=id_vars,
        value_vars=METRIC_NAMES,
        var_name="metric",
        value_name="auroc",
    )
    agg = (
        long.groupby(["temperature", "metric"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            n_seeds=("auroc", "count"),
        )
        .reset_index()
    )
    return agg


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--runs_root", default=RUNS_ROOT)
    p.add_argument("--output", default=OUT_CSV)
    p.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip the validator step; compute AUROC only from items already labeled.",
    )
    p.add_argument(
        "--original_t05_dir",
        default=ORIGINAL_T05_DIR,
        help=(
            "Replace the (T=0.5, seed=42) cell with prompt-matched items from "
            "this original paper-run directory. Set to '' to disable."
        ),
    )
    args = p.parse_args()

    suites = discover_suites(args.runs_root)
    if not suites:
        print(f"No matching suites in {args.runs_root}")
        return
    print(f"Found {len(suites)} suites under {args.runs_root}")

    rows = []
    for name, t, s in suites:
        suite_dir = os.path.join(args.runs_root, name)
        is_og_cell = (
            args.original_t05_dir
            and abs(t - OG_T) < 1e-6
            and s == OG_SEED
            and os.path.isdir(args.original_t05_dir)
        )

        if is_og_cell:
            # Use the original paper-run generations for this cell instead of
            # the regenerated sensitivity-sweep ones (same shuffle, same seed
            # → same 500 prompts, already validated).
            target_prompts = collect_prompts(suite_dir)
            print(
                f"\n{name}: substituting with prompt-matched items from "
                f"{args.original_t05_dir}"
            )
            res = auroc_row_from_original(args.original_t05_dir, target_prompts)
            source = "original"
        else:
            if not args.no_validate:
                print(f"\nValidating {name} ...")
                ensure_labels(suite_dir)
            res = auroc_row(suite_dir)
            source = "sweep"

        if res is None:
            print(f"{name}: no labeled samples, skipping.")
            continue
        aurocs, n, acc = res
        print(f"{name}: n={n}, acc={acc:.3f}, source={source}")
        rows.append(
            {
                "temperature": t,
                "seed": s,
                "source": source,
                "n_samples": n,
                "accuracy": acc,
                **aurocs,
            }
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = (
        pd.DataFrame(rows).sort_values(["temperature", "seed"]).reset_index(drop=True)
    )
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} per-(T,seed) rows to {args.output}")

    if not df.empty:
        agg = aggregate_across_seeds(df)
        agg_path = args.output.replace(".csv", "_by_temperature.csv")
        agg.to_csv(agg_path, index=False)
        print(f"Saved {len(agg)} aggregated rows to {agg_path}")

        print("\nPer-(T,seed):")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nAggregated over seeds:")
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
