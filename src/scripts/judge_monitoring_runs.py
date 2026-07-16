from src.engine.judge_monitoring_scenarios import evaluate_suite

SUITES = [
    # phi3-3b
    "phi3-3b-wildbench-test",
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
