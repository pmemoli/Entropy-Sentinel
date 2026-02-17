import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler

import joblib
import argparse

from typing import Literal

Model = Literal["logistic_regression", "random_forest", "neural_network"]

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
]


def run_training(
    train_suites: list[str],
    model: Model,
    model_name: str,
    balance_classes: bool = True,
    calibrate: bool = False,
    feature_subset: list[str] | None = None,
):
    X_list = []
    y_list = []
    for suite in train_suites:
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

            y_list.append(float(item["success"]))

    scaler = StandardScaler()
    X = torch.stack(X_list).float().numpy()
    X = scaler.fit_transform(X)
    y = np.array(y_list, dtype=np.float32)
    X, y = shuffle(X, y, random_state=42)

    min_class_count = min(np.bincount(y.astype(int)))

    if model == "logistic_regression":
        base_model = LogisticRegression(
            solver="liblinear",
            max_iter=10000,
            random_state=42,
            class_weight="balanced" if balance_classes else None,
        )
        param_grid = {"C": [0.5, 2.0, 10.0], "penalty": ["l1"]}

    elif model == "random_forest":
        base_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced_subsample" if balance_classes else None,
        )
        param_grid = {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
        }

    else:
        if balance_classes:
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)  # type: ignore

        base_model = MLPClassifier(
            max_iter=10000,
            random_state=42,
            activation="relu",
            early_stopping=True,
            alpha=0.001,
        )
        param_grid = {
            "hidden_layer_sizes": [
                (5,),
                (8,),
                (10,),
                (15,),
                (20,),
                (8, 4),
                (10, 5),
                (15, 8),
            ],
        }

    cv_folds = min(5, min_class_count)
    if cv_folds < 2:
        print("Not enough samples to perform cross-validation.")
        return

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_folds,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    print(grid_search.best_params_)

    trained_model = grid_search.best_estimator_

    if calibrate:
        cv_folds = min(5, min_class_count)
        calibrator = CalibratedClassifierCV(
            estimator=trained_model, method="isotonic", cv=cv_folds
        )
        calibrator.fit(X, y)
        trained_model = calibrator

    joblib.dump(trained_model, f"src/data/models/{model_name}.joblib")
    joblib.dump(scaler, f"src/data/models/{model_name}_scaler.joblib")

    print(f"Training complete. Model and Scaler saved.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train logistic regression on extracted features",
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
            "neural_network",
        ],
        required=True,
        help="Type of model to train",
    )
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Whether to balance classes during training",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Whether to calibrate the model using isotonic regression",
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
            balance_classes=args.balance_classes,
            calibrate=args.calibrate,
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {str(e)}")


if __name__ == "__main__":
    main()
