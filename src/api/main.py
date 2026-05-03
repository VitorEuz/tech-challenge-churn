from fastapi import FastAPI
from pydantic import BaseModel

from src.models.predict_model import predict as predict_churn


app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn usando MLP PyTorch",
    version="0.1.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    return predict_churn(customer.model_dump())