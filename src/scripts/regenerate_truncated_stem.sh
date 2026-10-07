MAX_LENGTH=16384

models=(
    "meta-llama/Llama-3.1-8B-Instruct" "llama3-8b"
    "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-8b"
    "openai/gpt-oss-20b" "oss-20b"
    "Qwen/Qwen3-4B-Instruct-2507" "qwen3-4b"
    "google/gemma-3-12b-it" "gemma3-12b"
    "microsoft/Phi-3.5-mini-instruct" "phi3-3b"
)

datasets=(
    "gsm8k" "gsm"
    "gsm8ksymbolic" "gsmsymbolic"
    "svamp" "svamp"
    "mathhendrycks" "mathhendrycks"
    "livemathbench" "livemathbench"
    "gpqa" "gpqa"
    "scibench" "scibench"
)

for ((m = 0; m < ${#models[@]}; m+=2)); do
    model_name=${models[m]}
    llm=${models[m+1]}

    for ((d = 0; d < ${#datasets[@]}; d+=2)); do
        dataset_name=${datasets[d]}
        suite="${llm}-${datasets[d+1]}-test"

        echo "Processing suite: ${suite}"

        uv run python -m src.scripts.strip_truncated_stem_runs \
          --suites "${suite}"

        uv run python -m src.engine.run_stem_scenarios \
          --dataset_name "${dataset_name}" \
          --split "test" \
          --model_name "${model_name}" \
          --suite "${suite}" \
          --result_path "./src/data/runs" \
          --max_length "${MAX_LENGTH}"

        echo "Completed regeneration for ${suite}."
    done
done
