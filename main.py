import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:7000")

def train_model(C, solver, max_iter):
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return model, accuracy, precision, recall


def main():
    # mlflow.set_experiment("classification_experiment")

    # Parameter grid for tuning
    param_grid = [
        {'C': 0.01, 'solver': 'lbfgs', 'max_iter': 100},
        {'C': 0.01, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 0.01, 'solver': 'liblinear', 'max_iter': 100},
        {'C': 0.01, 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 0.1, 'solver': 'lbfgs', 'max_iter': 100},
        {'C': 0.1, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 0.1, 'solver': 'liblinear', 'max_iter': 100},
        {'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 1, 'solver': 'lbfgs', 'max_iter': 100},
        {'C': 1, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 1, 'solver': 'liblinear', 'max_iter': 100},
        {'C': 1, 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 10, 'solver': 'lbfgs', 'max_iter': 100},
        {'C': 10, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 10, 'solver': 'liblinear', 'max_iter': 100},
        {'C': 10, 'solver': 'liblinear', 'max_iter': 1000},
    ]

    best_accuracy = 0
    best_run_id = None

    mlflow.set_experiment("Logistic Regression classification")

    for params in param_grid:
        with mlflow.start_run():
            model, accuracy, precision, recall = train_model(**params)

            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall
            })

            mlflow.set_tag("Training Info", "Basic LR model for data generated by the make_classification function")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = mlflow.active_run().info.run_id

    # Register the best model
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "best_model")


if __name__ == "__main__":
    main()