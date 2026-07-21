from src.engine.judge_monitoring_scenarios import evaluate_suite

SUITES = [
    # phi3-3b
    "phi3-3b-wildbench-test",
    # qwen3-4b
    "qwen3-4b-wildbench-test",
    # qwen3-8b
    "qwen3-8b-wildbench-test",
    # ministral3-3b
    "ministral3-3b-wildbench-test",
    # ministral3-8b
    "ministral3-8b-wildbench-test",
    # llama3-8b
    "llama3-8b-wildbench-test",
    # gemma3-4b
    "gemma3-4b-wildbench-test",
    # gemma3-12b
    "gemma3-12b-wildbench-test",
    # oss-20b
    "oss-20b-wildbench-test",
]


def main():
    for suite in SUITES:
        print(f"Processing suite: {suite}")

        # A missing or malformed suite shouldn't take down the whole sweep.
        try:
            evaluate_suite(suite)
        except Exception as e:
            print(f"  FAILED {suite}: {e}")
            continue

        print(f"Completed evaluating {suite}.")


if __name__ == "__main__":
    main()
