from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.train_model import MLP


MODEL_PATH = Path("models/best_mlp.pth")
SCALER_PATH = Path("models/scaler.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")


def load_model_and_tools():
    feature_names = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    model = MLP(len(feature_names))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    return model, scaler, feature_names


def preprocess_single_input(data: dict, feature_names):
    df = pd.DataFrame([data])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = pd.get_dummies(df)

    # garantir mesmas colunas do treino
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


def predict(data: dict):
    model, scaler, feature_names = load_model_and_tools()

    df = preprocess_single_input(data, feature_names)

    X = scaler.transform(df)

    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        prob = model(X_tensor).item()

    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "churn_probability": round(prob, 4),
        "label": "Churn" if pred == 1 else "No Churn",
    }