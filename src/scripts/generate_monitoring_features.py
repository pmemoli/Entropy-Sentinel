from src.engine.generate_monitoring_features import run

SUITES = [
    "phi3-3b-wildbench-test",
    "phi3-3b-mtbench-test",
    "qwen3-4b-wildbench-test",
    "qwen3-8b-wildbench-test",
    "ministral3-3b-wildbench-test",
    "ministral3-8b-wildbench-test",
    "llama3-8b-wildbench-test",
    "gemma3-4b-wildbench-test",
    "gemma3-12b-wildbench-test",
    "oss-20b-wildbench-test",
]


def main():
    for suite in SUITES:
        print(f"Processing suite: {suite}")

        # A missing or malformed suite shouldn't take down the whole sweep.
        try:
            run(suite)
        except Exception as e:
            print(f"  FAILED {suite}: {e}")
            continue

        print(f"Completed generating features for {suite}.")


if __name__ == "__main__":
    main()
