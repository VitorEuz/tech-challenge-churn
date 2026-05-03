from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_raw_data
from src.features.preprocessing import preprocess_data


SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
THRESHOLD = 0.5


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("TechChallenge_Churn_Prediction")

    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

    X_test_tensor = torch.FloatTensor(X_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    input_dim = X_train.shape[1]
    model = MLP(input_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_loss = np.inf
    counter = 0

    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    best_model_path = models_path / "best_mlp.pth"
    scaler_path = models_path / "scaler.pkl"
    features_path = models_path / "feature_names.pkl"

    with mlflow.start_run(run_name="MLP_PyTorch_Final"):
        mlflow.log_param("model_type", "MLP PyTorch")
        mlflow.log_param("seed", SEED)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("dataset_name", "telco_churn.csv")
        mlflow.log_param("dataset_version", "v1.0_limpo_padronizado")

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                counter += 1

            if counter >= PATIENCE:
                print("Early Stopping acionado.")
                break

        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        with torch.no_grad():
            y_prob = model(X_test_tensor).numpy().ravel()

        y_pred = (y_prob >= THRESHOLD).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        custo_fp = 20
        custo_fn = 200
        custo_total = (fp * custo_fp) + (fn * custo_fn)

        print("=== RESULTADOS MLP ===")
        print(f"AUC ROC: {auc:.4f}")
        print(f"PR AUC : {pr_auc:.4f}")
        print(f"F1     : {f1:.4f}")
        print(f"ACC    : {acc:.4f}")

        print("=== MATRIZ DE CONFUSÃO ===")
        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"TN: {tn}")

        print("=== TRADE-OFF DE CUSTO ===")
        print(f"Custo FP: R$ {fp * custo_fp}")
        print(f"Custo FN: R$ {fn * custo_fn}")
        print(f"Custo Total: R$ {custo_total}")

        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_names, features_path)

        mlflow.log_metric("best_loss", best_loss)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("true_positives", tp)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("false_negatives", fn)
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("cost_fp", fp * custo_fp)
        mlflow.log_metric("cost_fn", fn * custo_fn)
        mlflow.log_metric("cost_total", custo_total)

        mlflow.pytorch.log_model(model, "modelo_mlp")
        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(features_path))

    print(f"Modelo salvo em: {best_model_path}")
    print(f"Scaler salvo em: {scaler_path}")
    print(f"Features salvas em: {features_path}")


if __name__ == "__main__":
    train_model()