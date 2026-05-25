#!/usr/bin/env bash
# W/AUROC sensitivity sweep on MATH-test (Phi-3.5-Mini).
# Two axes in one grid:
#   - temperature: deployment-low to above-paper
#   - seed:        sampling-noise at fixed temperature
# Each (temperature, seed) cell shares the same 500 MATH items because
# store_activations.py pins item selection to a fixed RNG independent of --seed.
set -euo pipefail

MODEL="microsoft/Phi-3.5-mini-instruct"
DATASET="mathhendrycks"
SPLIT="test"
MAX_SAMPLES=500
MAX_LENGTH=4096
RESULT_PATH="./src/data/sensitivity"

TEMPERATURES=(0.3 0.5 0.7 1.0)
SEEDS=(42 43 44 45 46)

for temperature in "${TEMPERATURES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    suite="phi3-3b-math-sensitivity-t${temperature}-s${seed}"
    echo "=== ${suite} (T=${temperature}, seed=${seed}) ==="

    python3 -m src.engine.store_activations \
      --dataset_name "${DATASET}" \
      --split "${SPLIT}" \
      --model_name "${MODEL}" \
      --suite "${suite}" \
      --result_path "${RESULT_PATH}" \
      --max_length "${MAX_LENGTH}" \
      --max_samples "${MAX_SAMPLES}" \
      --temperature "${temperature}" \
      --seed "${seed}"

    echo "=== done ${suite} ==="
  done
done
