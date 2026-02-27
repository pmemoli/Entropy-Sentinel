import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

import joblib
import argparse


feature_names = [
    "mean",
    "std",
    "max",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "skewness",
    "kurtosis",
    "se_sum",
    "nll_avg",
    "nll_max",
    "nll_sum",
    "lntp",
    "mtp",
    "ppl",
]


def run_training(
    train_suites: list[str],
    model_name: str,
    feature: str = "mean",
):
    index = feature_names.index(feature)

    X_list = []
    y_list = []
    for suite in train_suites:
        data_path = f"src/data/features/{suite}.pt"
        raw_data = torch.load(data_path)

        if "test" in suite:
            split_index = int(0.8 * len(raw_data))
            np.random.seed(42)
            np.random.shuffle(raw_data)
            raw_data = raw_data[:split_index]

        for item in raw_data:
            X_list.append(item["features:"][[index]])
            y_list.append(float(item["success"]))

    scaler = StandardScaler()
    X = torch.stack(X_list).float().numpy()
    X = scaler.fit_transform(X)
    y = np.array(y_list, dtype=np.float32)
    X, y = shuffle(X, y, random_state=42)

    # Platt calibration
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=10000,
        random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, f"src/data/models/{model_name}.joblib")
    joblib.dump(scaler, f"src/data/models/{model_name}_scaler.joblib")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Platt calibration baseline on extracted features",
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
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        run_training(
            train_suites=args.train_suites,
            model_name=args.model_name,
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {str(e)}")


if __name__ == "__main__":
    main()
