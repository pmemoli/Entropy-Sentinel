presets=(
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
