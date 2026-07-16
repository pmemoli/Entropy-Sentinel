from src.engine.generate_stem_features import run

SUITES = [
    # phi3-3b
    "phi3-3b-livemathbench-test",
    "phi3-3b-gpqa-test",
    "phi3-3b-gsm-test",
    "phi3-3b-gsmsymbolic-test",
    "phi3-3b-mathhendrycks-test",
    "phi3-3b-matscibench-test",
    "phi3-3b-olympiadbench-test",
    "phi3-3b-scibench-test",
    "phi3-3b-svamp-test",
    "phi3-3b-theoremqa-test",
    # qwen3-4b
    "qwen3-4b-gpqa-test",
    "qwen3-4b-gsm-test",
    "qwen3-4b-gsmsymbolic-test",
    "qwen3-4b-mathhendrycks-test",
    "qwen3-4b-olympiadbench-test",
    "qwen3-4b-scibench-test",
    "qwen3-4b-svamp-test",
    "qwen3-4b-theoremqa-test",
    "qwen3-4b-matscibench-test",
    "qwen3-4b-livemathbench-test",
    # qwen3-8b
    "qwen3-8b-gpqa-test",
    "qwen3-8b-gsm-test",
    "qwen3-8b-gsmsymbolic-test",
    "qwen3-8b-mathhendrycks-test",
    "qwen3-8b-olympiadbench-test",
    "qwen3-8b-scibench-test",
    "qwen3-8b-svamp-test",
    "qwen3-8b-theoremqa-test",
    "qwen3-8b-matscibench-test",
    "qwen3-8b-livemathbench-test",
    # ministral3-3b
    "ministral3-3b-gpqa-test",
    "ministral3-3b-gsm-test",
    "ministral3-3b-gsmsymbolic-test",
    "ministral3-3b-mathhendrycks-test",
    "ministral3-3b-matscibench-test",
    "ministral3-3b-olympiadbench-test",
    "ministral3-3b-scibench-test",
    "ministral3-3b-svamp-test",
    "ministral3-3b-theoremqa-test",
    "ministral3-3b-livemathbench-test",
    # ministral3-8b
    "ministral3-8b-gpqa-test",
    "ministral3-8b-gsm-test",
    "ministral3-8b-gsmsymbolic-test",
    "ministral3-8b-mathhendrycks-test",
    "ministral3-8b-olympiadbench-test",
    "ministral3-8b-scibench-test",
    "ministral3-8b-svamp-test",
    "ministral3-8b-theoremqa-test",
    "ministral3-8b-matscibench-test",
    "ministral3-8b-livemathbench-test",
    # oss-20b
    "oss-20b-gpqa-test",
    "oss-20b-gsm-test",
    "oss-20b-gsmsymbolic-test",
    "oss-20b-mathhendrycks-test",
    "oss-20b-olympiadbench-test",
    "oss-20b-scibench-test",
    "oss-20b-svamp-test",
    "oss-20b-theoremqa-test",
    "oss-20b-matscibench-test",
    "oss-20b-livemathbench-test",
    # llama3-8b
    "llama3-8b-gpqa-test",
    "llama3-8b-gsm-test",
    "llama3-8b-gsmsymbolic-test",
    "llama3-8b-math-test",
    "llama3-8b-mathhendrycks-test",
    "llama3-8b-olympiadbench-test",
    "llama3-8b-scibench-test",
    "llama3-8b-svamp-test",
    "llama3-8b-theoremqa-test",
    "llama3-8b-matscibench-test",
    "llama3-8b-livemathbench-test",
    # gemma3-4b
    "gemma3-4b-gpqa-test",
    "gemma3-4b-gsm-test",
    "gemma3-4b-gsmsymbolic-test",
    "gemma3-4b-mathhendrycks-test",
    "gemma3-4b-olympiadbench-test",
    "gemma3-4b-scibench-test",
    "gemma3-4b-svamp-test",
    "gemma3-4b-theoremqa-test",
    "gemma3-4b-matscibench-test",
    "gemma3-4b-livemathbench-test",
    # gemma3-12b
    "gemma3-12b-gpqa-test",
    "gemma3-12b-gsm-test",
    "gemma3-12b-gsmsymbolic-test",
    "gemma3-12b-mathhendrycks-test",
    "gemma3-12b-olympiadbench-test",
    "gemma3-12b-scibench-test",
    "gemma3-12b-svamp-test",
    "gemma3-12b-theoremqa-test",
    "gemma3-12b-matscibench-test",
    "gemma3-12b-livemathbench-test",
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
