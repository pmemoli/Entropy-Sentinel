export VLLM_USE_V1=0

presets=(
    # "arenahard" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-3b-arenahard-test"
    # "mtbench" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-3b-mtbench-test"
    "wildbench" "test" "mistralai/Ministral-8B-Instruct-2410" "ministral3-8b-wildbench-test"
    "wildbench" "test" "google/gemma-3-12b-it" "gemma3-12b-wildbench-test"
    "wildbench" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-3b-wildbench-test"
)

for ((i = 0; i < ${#presets[@]}; i+=4)); do
    dataset_name=${presets[i]}
    split=${presets[i+1]}
    model_name=${presets[i+2]}
    suite=${presets[i+3]}

    echo "Processing suite: ${suite}"

    uv run python -m src.engine.run_monitoring_scenarios \
      --dataset_name "${dataset_name}" \
      --split "${split}" \
      --model_name "${model_name}" \
      --suite "${suite}" \
      --result_path "./src/data/runs" \
      --max_length 16384

    echo "Completed storing activations for ${suite}."
done
