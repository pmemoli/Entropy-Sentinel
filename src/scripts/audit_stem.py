import argparse
import csv
import os
import random
import time

import torch

DEFAULT_OUT = "src/data/audit.csv"
RUNS_ROOT = "src/data/runs"

_CHOICES = {
    "y": "agree",
    "n": "disagree",
    "s": "skip",
}


def is_stem_batch(batch):
    """Only STEM runs carry a judged `success` label on a flat item. Monitoring
    runs nest their generations in messages[] and are scored 1-10, so there is
    nothing here to audit."""
    if not batch:
        return False

    item = batch[0]
    return "generation" in item and "success" in item


def list_suite_files(runs_root, suites):
    if not suites or suites == ["all"]:
        if not os.path.isdir(runs_root):
            return []
        suites = sorted(os.listdir(runs_root))
    out = []
    for suite in suites:
        d = os.path.join(runs_root, suite)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".pt"):
                continue
            try:
                batch = torch.load(
                    os.path.join(d, fname),
                    map_location="cpu",
                    weights_only=False,
                )
                if not is_stem_batch(batch):
                    print(f"  Skipping non-STEM: {suite}/{fname}")
                    continue

                out.append((suite, fname, len(batch)))
            except Exception as e:
                print(f"  WARN: skipping {suite}/{fname}: {e}")
    return out


def load_existing(path):
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            seen.add((r["suite"], r["file"], int(r["idx"])))
    return seen


def append_row(path, row):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "suite",
                "file",
                "idx",
                "validator_label",
                "user_label",
                "timestamp",
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def sample_targets(suite_files, n, seed, exclude):
    pool = [
        (s, f, i)
        for (s, f, n_items) in suite_files
        for i in range(n_items)
        if (s, f, i) not in exclude
    ]
    random.Random(seed).shuffle(pool)
    picked = pool[:n]
    # Group by (suite, file) so each batch is loaded once.
    picked.sort(key=lambda t: (t[0], t[1]))
    return picked


def prompt(validator_label: bool):
    print(f"  validator says: {'CORRECT' if validator_label else 'WRONG'}")
    print("  [y] agree  [n] disagree  [s] skip  [q] save & quit")
    while True:
        c = input("> ").strip().lower()
        if c in _CHOICES or c == "q":
            return c
        print("  please type y/n/s/q")


def summarize(path):
    if not os.path.exists(path):
        return
    counts = {"agree": 0, "disagree": 0, "skip": 0}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            counts[r["user_label"]] = counts.get(r["user_label"], 0) + 1
    decided = counts["agree"] + counts["disagree"]
    print(
        f"\nAudit total: {counts['agree']} agree, {counts['disagree']} disagree, "
        f"{counts['skip']} skip"
    )
    if decided:
        print(
            f"Agreement rate (excluding skips): "
            f"{counts['agree']/decided:.3f} ({counts['agree']}/{decided})"
        )


def run(args):
    print(f"Scanning {args.runs_root}...")
    suite_files = list_suite_files(args.runs_root, args.suites)
    if not suite_files:
        print("No suites found.")
        return
    total = sum(s[2] for s in suite_files)
    print(
        f"  {total} samples across {len(suite_files)} files in "
        f"{len({s[0] for s in suite_files})} suites."
    )

    existing = load_existing(args.output)
    targets = sample_targets(suite_files, args.n, args.seed, existing)
    print(
        f"  Auditing {len(targets)} new instances "
        f"({len(existing)} already in {args.output})."
    )

    loaded_key = None
    batch = None
    try:
        for i, (suite, fname, idx) in enumerate(targets):
            if (suite, fname) != loaded_key:
                batch = torch.load(
                    os.path.join(args.runs_root, suite, fname),
                    map_location="cpu",
                    weights_only=False,
                )
                loaded_key = (suite, fname)
            item = batch[idx]
            print("\n" + "=" * 72)
            print(f"[{i+1}/{len(targets)}] {suite}  {fname}  idx={idx}")
            print("=" * 72)
            print(f"\nQUESTION:\n{item['prompt']}\n")
            print(f"REFERENCE:\n{item['reference']}\n")
            print(f"GENERATION:\n{item['generation']}\n")
            choice = prompt(bool(item["success"]))
            if choice == "q":
                print("Quitting; progress is already saved.")
                break
            append_row(
                args.output,
                {
                    "suite": suite,
                    "file": fname,
                    "idx": idx,
                    "validator_label": int(bool(item["success"])),
                    "user_label": _CHOICES[choice],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; progress is already saved.")

    summarize(args.output)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Manual audit of validator success labels on STEM suites.",
    )
    p.add_argument("--runs_root", default=RUNS_ROOT)
    p.add_argument(
        "--suites",
        nargs="+",
        default=["all"],
        help="STEM suite dirs under runs_root, or 'all'.",
    )
    p.add_argument(
        "--n", type=int, default=600, help="New instances to audit."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=DEFAULT_OUT)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
