import os
from src.engine.train_classifier import run_training

llms = [
    "qwen3-8b",
    "phi3-3b",
    "qwen3-4b",
    "ministral3-3b",
    "ministral3-8b",
    "llama3-8b",
    "gemma3-4b",
    "gemma3-12b",
    "oss-20b",
]

configurations = {
    "rf_extremes": {
        "model": "random_forest",
        "benchmarks": ["gsm-test", "olympiadbench-test"],
        "calibrate": True,
        "balance_classes": True,
    },
    "rf_intermediate": {
        "model": "random_forest",
        "benchmarks": ["svamp-test", "mathhendrycks-test", "gpqa-test"],
        "calibrate": True,
        "balance_classes": True,
    },
    "logistic_balanced": {
        "model": "logistic_regression",
        "benchmarks": ["gsmsymbolic-test", "matscibench-test"],
        "calibrate": True,
        "balance_classes": True,
    },
    "nn_flexible": {
        "model": "neural_network",
        "benchmarks": ["gpqa-test", "matscibench-test", "gsm-test"],
        "calibrate": True,
        "balance_classes": True,
    },
}

MODEL_DIR = "src/data/models/"
existing_files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []

skipped = 0
completed = 0

total_tasks = len(llms) * len(configurations)
current = 0

for llm in llms:
    for config_name, settings in configurations.items():
        current += 1

        joblib_name = f"{config_name}_{llm}"
        filename = f"{joblib_name}.joblib"

        if filename in existing_files:
            print(f"[{current}/{total_tasks}] Skipping {joblib_name} (exists)")
            skipped += 1
            continue

        print(f"[{current}/{total_tasks}] Training {joblib_name}...")

        train_suites = [f"{llm}-{bench}" for bench in settings["benchmarks"]]

        try:
            run_training(
                train_suites=train_suites,
                model=settings["model"],
                model_name=joblib_name,
                balance_classes=settings["balance_classes"],
                calibrate=settings["calibrate"],
            )
            completed += 1
        except Exception as e:
            print(f"Failed to train {joblib_name}: {e}")

print(f"\n--- Process Complete ---")
print(f"Completed: {completed}")
print(f"Skipped: {skipped}")
