"""Smoke test for src/engine/run_monitoring_scenarios.py.

Runs the real runner over 5 wildbench samples on Phi-3.5-mini, into a throwaway
suite so it can't touch phi3-3b-wildbench-test, then checks the stored results
look like usable monitoring runs. Needs a GPU — vLLM loads the model for real.

    uv run python -m src.informal          # cleans up the throwaway suite
    uv run python -m src.informal --keep   # leaves it for inspection
"""

import argparse
import glob
import shutil
import subprocess
import sys

import torch

DATASET = "wildbench"
MODEL = "microsoft/Phi-3.5-mini-instruct"
SUITE = "phi3-3b-wildbench-smoketest"
RESULT_PATH = "./src/data/runs"
N_SAMPLES = 5

SUITE_DIR = f"{RESULT_PATH}/{SUITE}"


def run_scenarios():
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.engine.run_monitoring_scenarios",
            "--dataset_name",
            DATASET,
            "--split",
            "test",
            "--model_name",
            MODEL,
            "--suite",
            SUITE,
            "--result_path",
            RESULT_PATH,
            "--max_length",
            "4096",
            "--max_samples",
            str(N_SAMPLES),
        ],
        text=True,
    )


def check_results():
    """Returns a list of problems; empty means the run looks healthy."""
    problems = []

    files = glob.glob(f"{SUITE_DIR}/*.pt")
    if not files:
        return [f"no result files written to {SUITE_DIR}"]

    results = [result for file in files for result in torch.load(file)]

    if len(results) != N_SAMPLES:
        problems.append(f"stored {len(results)} results, expected {N_SAMPLES}")

    for i, result in enumerate(results):
        messages = result["messages"]
        if not messages:
            problems.append(f"result {i}: no messages")
            continue

        roles = [message["role"] for message in messages]
        if roles != ["user", "assistant"] * (len(roles) // 2):
            problems.append(f"result {i}: roles not alternating user/assistant: {roles}")

        for j, message in enumerate(messages):
            if message["role"] != "assistant":
                continue

            profile = message["entropy_profile"]
            logprobs = message["selected_logprobs"]
            n_tokens = len(message["token_ids"])

            if n_tokens == 0:
                problems.append(f"result {i} message {j}: empty generation")
                continue
            # One entropy value and one logprob per generated token is what the
            # feature extraction downstream assumes.
            if len(profile) != n_tokens:
                problems.append(
                    f"result {i} message {j}: {len(profile)} entropies for {n_tokens} tokens"
                )
            if len(logprobs) != n_tokens:
                problems.append(
                    f"result {i} message {j}: {len(logprobs)} logprobs for {n_tokens} tokens"
                )
            if not torch.isfinite(profile).all():
                problems.append(f"result {i} message {j}: non-finite entropy values")
            if (profile < 0).any():
                problems.append(f"result {i} message {j}: negative entropy values")

    if not problems:
        turns = sum(
            1 for r in results for m in r["messages"] if m["role"] == "assistant"
        )
        print(f"\n{len(results)} results, {turns} assistant turns, all well-formed.")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the throwaway suite directory instead of deleting it.",
    )
    args = parser.parse_args()

    if glob.glob(f"{SUITE_DIR}/*.pt"):
        print(f"{SUITE_DIR} already has results; remove it first.")
        sys.exit(1)

    print(f"Running {DATASET} on {MODEL}, {N_SAMPLES} samples -> {SUITE_DIR}\n")

    try:
        result = run_scenarios()
        # run() swallows per-batch exceptions and still exits 0, so the exit
        # code alone doesn't tell us the generation actually worked.
        if result.returncode != 0:
            print(f"\nFAILED: runner exited {result.returncode}")
            sys.exit(1)

        problems = check_results()
    finally:
        if not args.keep:
            shutil.rmtree(SUITE_DIR, ignore_errors=True)

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)

    print("All good.")


if __name__ == "__main__":
    main()


