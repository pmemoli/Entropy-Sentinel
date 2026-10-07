import argparse
import glob
import torch

EOS_TOKENS = {
    "qwen3": {151645, 151643},
    "phi3": {32007, 32000},
    "gemma3": {106, 1},
    "llama3": {128009, 128001},
    "ministral3": {2},
    "oss": {200002, 200012, 199999},
}


def is_truncated(item, eos_tokens):
    sequences = item["sequences"]
    return len(sequences) == 0 or sequences[-1] not in eos_tokens


def strip_suite(suite, result_path, dry_run):
    eos_tokens = EOS_TOKENS[suite.split("-")[0]]
    total = removed = 0

    for path in sorted(glob.glob(f"{result_path}/{suite}/*.pt")):
        items = torch.load(path, weights_only=False)
        kept = [item for item in items if not is_truncated(item, eos_tokens)]
        total += len(items)
        removed += len(items) - len(kept)

        if not dry_run and len(kept) != len(items):
            torch.save(kept, path)

    print(f"{suite}: removed {removed} truncated of {total}")
    return removed


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--result_path", type=str, default="./src/data/runs")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    removed = sum(
        strip_suite(suite, args.result_path, args.dry_run)
        for suite in args.suites
    )
    print(f"Total removed: {removed}")


if __name__ == "__main__":
    main()
