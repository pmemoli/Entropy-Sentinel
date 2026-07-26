"""
Same AUROC table as ``metric_power.py``, but the entropy profile is recomputed
from full-vocab logits via teacher-forcing — no top-20 truncation.

For each (LLM, benchmark) pair, every saved (prompt + sampled token ids) tuple
is run through the HF model once. At each generation step we read the full-vocab
log-softmax and compute ``-Σ p log p`` over the full distribution, plus the
log-probability of the actually-sampled token. These full-vocab traces feed the
same metric/AUROC pipeline as ``metric_power.py``, so any difference is purely
the entropy source.

Per-pair full-vocab traces are cached at
  ``src/analysis/metric_predictive_power/cache/{llm}-{benchmark}.pt``
so re-running skips inference for pairs already done. Delete the cache file to
recompute.

Outputs (next to this file):
  - ``full_auroc_table.csv``  — AUROC table on full-vocab entropy
  - ``comparison.csv``        — per-(LLM, benchmark, metric) full − og delta

Defaults to the three models highlighted in the OG paper-table and the two
benchmarks the user is comparing (MATH + GSM8K).
"""

import argparse
import gc
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.analysis.metric_predictive_power.metric_power import (  # noqa: E402
    OUT_DIR,
    aurocs_from_metrics,
    iter_samples_from_dir,
    per_sample_metrics,
)

HF_IDS = {
    "phi3-3b": "microsoft/Phi-3.5-mini-instruct",
    "ministral3-8b": "mistralai/Ministral-3-8B-Instruct-2512",
    "oss-20b": "openai/gpt-oss-20b",
}

DEFAULT_LLMS = ["phi3-3b"]
DEFAULT_DATASETS = ["mathhendrycks-test", "gsm-test", "olympiadbench-test"]

CACHE_DIR = os.path.join(OUT_DIR, "cache")


@torch.no_grad()
def teacher_force_full_entropy(sample, model, tokenizer, device, max_model_len):
    """One teacher-forced forward pass; returns (entropy_profile, selected_logprobs)
    over the full vocabulary, both as cpu float32 numpy arrays."""
    prompt = sample["prompt"]
    gen_ids = list(sample["sequences"])
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )[0].tolist()
    full_ids = prompt_ids + gen_ids
    P = len(prompt_ids)
    T = len(gen_ids)
    if len(full_ids) > max_model_len:
        overflow = len(full_ids) - max_model_len
        print(
            f"  WARN: len {len(full_ids)} > max_model_len {max_model_len}; "
            f"truncating last {overflow} gen tokens for this sample"
        )
        full_ids = full_ids[:max_model_len]
        T = len(full_ids) - P
        gen_ids = gen_ids[:T]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, use_cache=False).logits[0]
    gen_logits = logits[P - 1 : P - 1 + T].float()
    log_probs = F.log_softmax(gen_logits, dim=-1)
    probs = torch.exp(log_probs)

    profile = -(probs * log_probs).sum(dim=-1)
    gen_ids_t = torch.tensor(gen_ids, device=device, dtype=torch.long)
    selected_lp = log_probs.gather(1, gen_ids_t.view(-1, 1)).squeeze(1)
    return profile.float().cpu().numpy(), selected_lp.float().cpu().numpy()


def cache_path(llm, dataset):
    return os.path.join(CACHE_DIR, f"{llm}-{dataset}.pt")


def collect_full_entropy_for_suite(
    llm, dataset, runs_root, device, max_model_len, checkpoint_every,
):
    cp = cache_path(llm, dataset)
    if os.path.exists(cp):
        cached = torch.load(cp, map_location="cpu", weights_only=False)
        print(f"  cached at {cp} (n={len(cached['profiles'])})")
        return cached

    if llm not in HF_IDS:
        print(f"  no HF id for {llm}, skipping")
        return None

    suite_dir = os.path.join(runs_root, f"{llm}-{dataset}")
    samples = list(iter_samples_from_dir(suite_dir))
    if not samples:
        print(f"  no samples in {suite_dir}")
        return None

    hf_id = HF_IDS[llm]
    print(f"  loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, trust_remote_code=False
        )
        .to(device)
        .eval()
    )

    profiles, lps, labels = [], [], []
    start = time.time()
    for i, sample in enumerate(samples):
        try:
            prof, lp = teacher_force_full_entropy(
                sample, model, tokenizer, device, max_model_len
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  OOM on sample {i} (gen_len={len(sample['sequences'])}); skipping")
            continue
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}; skipping")
            continue
        profiles.append(prof)
        lps.append(lp)
        labels.append(0 if sample["success"] else 1)
        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (len(samples) - (i + 1)) / max(rate, 1e-9)
            print(f"  [{i+1}/{len(samples)}] {rate:.2f} samp/s  ETA {eta/60:.1f} min")
        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            torch.save(
                {"profiles": profiles, "selected_logprobs": lps, "labels": labels},
                cp + ".ckpt",
            )
        if (i + 1) % 16 == 0:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    payload = {"profiles": profiles, "selected_logprobs": lps, "labels": labels}
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    torch.save(payload, cp)
    if os.path.exists(cp + ".ckpt"):
        os.remove(cp + ".ckpt")
    return payload


def build_full_table(llms, datasets, runs_root, device, max_model_len, checkpoint_every):
    rows = []
    for llm in llms:
        for ds in datasets:
            print(f"\n=== {llm}-{ds} ===")
            data = collect_full_entropy_for_suite(
                llm, ds, runs_root, device, max_model_len, checkpoint_every
            )
            if not data or not data["profiles"]:
                continue
            per_sample = [
                per_sample_metrics(p, lp)
                for p, lp in zip(data["profiles"], data["selected_logprobs"])
            ]
            aurocs = aurocs_from_metrics(per_sample, data["labels"])
            rows.append(
                {"LLM": llm, "Benchmark": ds, "n_samples": len(per_sample), **aurocs}
            )
    return pd.DataFrame(rows)


def write_comparison(full_df: pd.DataFrame, og_csv: str, out_csv: str):
    if not os.path.exists(og_csv):
        print(f"OG table not found at {og_csv}; skipping comparison")
        return
    og = pd.read_csv(og_csv)
    key = ["LLM", "Benchmark"]
    metric_cols = [c for c in full_df.columns if c not in key + ["n_samples"]]
    merged = full_df.merge(og, on=key, suffixes=("_full", "_og"))
    if merged.empty:
        print("No overlapping (LLM, Benchmark) rows between full and og; nothing to compare")
        return
    diff = merged[key].copy()
    for m in metric_cols:
        diff[m] = merged[f"{m}_full"] - merged[f"{m}_og"]
    diff.to_csv(out_csv, index=False)
    print(f"\nSaved comparison (full − og) to {out_csv}")
    print(diff.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Recompute the metric-predictive-power AUROC table using full-vocab "
            "entropy from teacher-forced logits, and diff vs the OG (top-20) table."
        ),
    )
    p.add_argument("--runs_root", default="src/data/runs")
    p.add_argument("--llms", nargs="+", default=DEFAULT_LLMS)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument(
        "--output", default=os.path.join(OUT_DIR, "full_auroc_table.csv")
    )
    p.add_argument(
        "--og_csv", default=os.path.join(OUT_DIR, "og_auroc_table.csv")
    )
    p.add_argument(
        "--comparison", default=os.path.join(OUT_DIR, "comparison.csv")
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = build_full_table(
        args.llms,
        args.datasets,
        args.runs_root,
        args.device,
        args.max_model_len,
        args.checkpoint_every,
    )
    df.to_csv(args.output, index=False)
    print(f"\nSaved full-vocab AUROC table ({len(df)} rows) to {args.output}")
    if not df.empty:
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    write_comparison(df, args.og_csv, args.comparison)


if __name__ == "__main__":
    main()
