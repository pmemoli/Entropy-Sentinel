# export VLLM_USE_V1=0

presets=(
    "wildbench" "test" "Qwen/Qwen3-8B" "qwen3-8b-wildbench-test"
    "wildbench" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-wildbench-test"
    "wildbench" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-8b-wildbench-test"
    "wildbench" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-4b-wildbench-test"
    "wildbench" "test" "google/gemma-3-4b-it" "gemma3-4b-wildbench-test"
    "wildbench" "test" "google/gemma-3-12b-it" "gemma3-12b-wildbench-test"
    "wildbench" "test" "openai/gpt-oss-20b" "oss-20b-wildbench-test"
    "wildbench" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-3b-wildbench-test"
    "wildbench" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-8b-wildbench-test"
)

for ((i = 0; i < ${#presets[@]}; i+=4)); do
    dataset_name=${presets[i]}
    split=${presets[i+1]}
    model_name=${presets[i+2]}
    suite=${presets[i+3]}

    echo "Processing suite: ${suite}"

    # Qwen3-8B's native context is 40960 (max_position_embeddings); 50000
    # exceeds it and would need VLLM_ALLOW_LONG_MAX_MODEL_LEN (RoPE -> NaN).
    if [[ "${model_name}" == "Qwen/Qwen3-8B" ]]; then
        max_length=40960
    else
        max_length=50000
    fi

    uv run python -m src.engine.run_monitoring_scenarios \
      --dataset_name "${dataset_name}" \
      --split "${split}" \
      --model_name "${model_name}" \
      --suite "${suite}" \
      --result_path "./src/data/runs" \
      --max_length "${max_length}" \
      --max_new_tokens 16384 \
      --batch_size 64


    echo "Completed storing activations for ${suite}."
done
