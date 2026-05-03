import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.data.load_data import load_raw_data
from src.features.preprocessing import preprocess_data


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)

        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, name)

    print(f"\n=== {name} ===")
    print(f"AUC ROC: {auc:.4f}")
    print(f"PR AUC : {pr_auc:.4f}")
    print(f"F1     : {f1:.4f}")
    print(f"ACC    : {acc:.4f}")


def run_baselines():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("TechChallenge_Churn_Prediction")

    df = load_raw_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df)

    evaluate_model(
        "Dummy",
        DummyClassifier(strategy="most_frequent"),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    evaluate_model(
        "LogisticRegression",
        LogisticRegression(max_iter=1000),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    evaluate_model(
        "RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    run_baselines()