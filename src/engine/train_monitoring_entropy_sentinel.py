import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import joblib
import argparse

from typing import Literal

Model = Literal["logistic_regression", "random_forest"]

# correctly indexed as stored
features_names = [
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

# The MT-bench judge rates on a 1-10 scale. ES regresses onto that score mapped
# to [0, 1], so a slice's mean prediction is comparable to the judge's own mean.
MIN_JUDGE_SCORE = 1
MAX_JUDGE_SCORE = 10


def normalize_judge_score(score: int) -> float:
    normalized = (score - MIN_JUDGE_SCORE) / (MAX_JUDGE_SCORE - MIN_JUDGE_SCORE)
    return float(np.clip(normalized, 0.0, 1.0))


class LogisticRegressionES:
    """Wraps the fitted linear layer so it predicts like the sklearn models."""

    def __init__(self, linear: torch.nn.Linear):
        self.linear = linear

    def predict(self, X):
        with torch.no_grad():
            logits = self.linear(torch.tensor(X, dtype=torch.float32))
            return torch.sigmoid(logits).squeeze(-1).numpy()


def fit_logistic_regression(X, y, steps: int = 2000, lr: float = 0.01):
    """Cross-entropy against the judge's normalized score. BCE takes soft
    targets, so the [0, 1] score is a valid target and the sigmoid keeps
    predictions bounded."""
    torch.manual_seed(42)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    linear = torch.nn.Linear(X_tensor.shape[1], 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(linear(X_tensor).squeeze(-1), y_tensor)
        loss.backward()
        optimizer.step()

    return LogisticRegressionES(linear)


def run_training(
    train_suites: list[str],
    model: Model,
    model_name: str,
    feature_subset: list[str] | None = None,
    suite_cache: dict | None = None,
):
    X_list = []
    y_list = []
    for suite in train_suites:
        if suite_cache is not None and suite in suite_cache:
            raw_data = suite_cache[suite]
        else:
            data_path = f"src/data/features/{suite}.pt"
            raw_data = torch.load(data_path)

        if "test" in suite:
            # split test suites into train and test halves (80-20 split)
            split_index = int(0.8 * len(raw_data))
            np.random.seed(42)
            np.random.shuffle(raw_data)

            raw_data = raw_data[:split_index]

        for item in raw_data:
            if feature_subset is not None:
                indices = [
                    features_names.index(name) for name in feature_subset
                ]
                selected_features = item["features:"][indices]
                X_list.append(selected_features)
            else:
                X_list.append(item["features:"])

            y_list.append(normalize_judge_score(item["judge_score"]))

    if not X_list:
        print("No judged samples found in the given suites.")
        return

    scaler = StandardScaler()
    X = torch.stack(X_list).float().numpy()
    X = scaler.fit_transform(X)
    y = np.array(y_list, dtype=np.float32)
    X, y = shuffle(X, y, random_state=42)

    if model == "logistic_regression":
        trained_model = fit_logistic_regression(X, y)

    else:
        base_model = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
        }

        cv_folds = min(5, len(y))
        if cv_folds < 2:
            print("Not enough samples to perform cross-validation.")
            return

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="r2",
            cv=cv_folds,
            verbose=0,
            n_jobs=-1,
        )

        grid_search.fit(X, y)

        print(grid_search.best_params_)

        trained_model = grid_search.best_estimator_

    joblib.dump(trained_model, f"src/data/models/{model_name}.joblib")
    joblib.dump(scaler, f"src/data/models/{model_name}_scaler.joblib")

    print(f"Training complete. Model and Scaler saved.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a regressor on extracted features against judge scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_suites", nargs="+", required=True, help="List of train suites"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "logistic_regression",
            "random_forest",
        ],
        required=True,
        help="Type of model to train",
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
            model=args.model,
            model_name=args.model_name,
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {str(e)}")


if __name__ == "__main__":
    main()
