#!/bin/bash

# Define the LLMs
llms=(
    "qwen3-8b"
    "phi3-3b"
    "qwen3-4b"
    "ministral3-3b"
    "ministral3-8b"
    "llama3-8b"
    "gemma3-4b"
    "gemma3-12b"
    "oss-20b"
)

# Define the features to iterate over
features=(
    "Max"
    "Mean"
    "SE_sum"
    "NLL_avg"
    "NLL_max"
    "NLL_sum"
    "LNTP"
    "MTP"
    "PPL"
)

echo "Starting calibration training for ${#llms[@]} models and ${#features[@]} features..."

for model in "${llms[@]}"; do
    for feature in "${features[@]}"; do
        
        # Construct the model name for the output file
        # e.g., phi3-3b_SE_sum_calibrator
        OUTPUT_NAME="${model}_${feature}_calibrator"
        
        echo "----------------------------------------------------------------"
        echo "Training: $model | Feature: $feature"
        echo "Output: $OUTPUT_NAME"
        
        # Run the training command
        # Note: We prefix the suite names with the model name to match your file structure
        python -m src.engine.train_calibrator \
            --train_suites "${model}-gsm-test" "${model}-olympiadbench-test" \
            --model_name "$OUTPUT_NAME" \
            --feature_subset "$feature"
            
        if [ $? -ne 0 ]; then
            echo "Error training $model with $feature. Continuing..."
        fi
    done
done

echo "All training runs complete."
