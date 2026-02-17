import torch
import numpy as np
import os
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import joblib
import argparse
from typing import Dict

# Master list of feature keys matching your metrics_data logic
FEATURE_NAMES = [
    "Max",
    "Mean",
    "STD",
    "Q10",
    "Q25",
    "Q50",
    "Q75",
    "Q90",
    "Skew",
    "Kurt",
    "SE_sum",
    "NLL_avg",
    "NLL_max",
    "NLL_sum",
    "LNTP",
    "MTP",
    "PPL",
]


def calculate_metrics_from_item(item: dict) -> np.ndarray:
    """
    Exact implementation of your metric logic for a single run item.
    """
    # 1. Extract raw data
    profile = item["entropy_profile"].to(torch.float32).numpy()
    selected_logprobs = item["selected_logprobs"]
    if isinstance(selected_logprobs, torch.Tensor):
        logprobs = selected_logprobs.to(torch.float32).numpy()
    else:
        logprobs = np.array(selected_logprobs, dtype=np.float32)

    # Pre-calculate NLL for the baselines
    nll = -logprobs

    # 2. Build the vector in the exact order of FEATURE_NAMES
    stats = [
        # Statistical moments
        np.max(profile),
        np.mean(profile),
        np.std(profile),
        np.percentile(profile, 10),
        np.percentile(profile, 25),
        np.percentile(profile, 50),
        np.percentile(profile, 75),
        np.percentile(profile, 90),
        skew(profile) if len(profile) > 2 else 0.0,
        kurtosis(profile) if len(profile) > 2 else 0.0,
        # Strong Baselines
        np.sum(profile),  # SE_sum (EAS)
        np.mean(nll),  # NLL_avg
        np.max(nll),  # NLL_max
        np.sum(nll),  # NLL_sum
        np.exp(np.mean(logprobs)),  # LNTP
        np.exp(np.min(logprobs)),  # MTP
        np.exp(-np.mean(logprobs)),  # PPL
    ]
    return np.array(stats)


def run_platt_training(
    train_suites: list[str],
    model_name: str,
    feature_subset: list[str] | None = None,
):
    X_list = []
    y_list = []

    for suite in train_suites:
        # FOLDER BASED LOADING (per your code snippet)
        tensor_dir = f"src/data/runs_with_logprobs/{suite}"
        if not os.path.exists(tensor_dir):
            print(f"Warning: Directory {tensor_dir} not found. skipping.")
            continue

        tensor_files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]

        for file in tensor_files:
            full_path = os.path.join(tensor_dir, file)
            # Use map_location='cpu' for safety
            raw_data = torch.load(full_path, map_location="cpu")

            for item in raw_data:
                # Calculate X
                features = calculate_metrics_from_item(item)
                X_list.append(features)

                # Calculate y: 1 = failure, 0 = success (per your snippet)
                is_failure = 0 if item["success"] else 1
                y_list.append(float(is_failure))

    if not X_list:
        print("No data loaded. Check your suite paths.")
        return

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    X, y = shuffle(X, y, random_state=42)

    features_to_train = feature_subset if feature_subset else FEATURE_NAMES
    platt_models: Dict[str, LogisticRegression] = {}

    for feature_name in features_to_train:
        if feature_name not in FEATURE_NAMES:
            print(f"Skipping unknown feature: {feature_name}")
            continue

        feature_idx = FEATURE_NAMES.index(feature_name)
        feature_values = X[:, feature_idx].reshape(-1, 1)

        # Platt scaling (Logistic Regression with low regularization)
        platt_model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        platt_model.fit(feature_values, y)
        platt_models[feature_name] = platt_model

        coef = platt_model.coef_[0][0]
        print(
            f"  {feature_name:10}: w={coef:.4f}, b={platt_model.intercept_[0]:.4f}"
        )

    # Ensure model directory exists
    os.makedirs("src/data/models", exist_ok=True)
    save_path = f"src/data/models/{model_name}_platt.joblib"
    joblib.dump(platt_models, save_path)
    print(f"\nSaved {len(platt_models)} models to {save_path}")

    return platt_models


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Platt scaling calibration models for each feature variable",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_suites", nargs="+", required=True, help="List of train suites"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name for the saved model file",
    )
    parser.add_argument(
        "--feature_subset",
        nargs="+",
        default=None,
        help="Optional subset of features to train on (default: all features)",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        run_platt_training(
            train_suites=args.train_suites,
            model_name=args.model_name,
            feature_subset=args.feature_subset,
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
