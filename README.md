# Tech Challenge - Churn Prediction

## Integrantes

- Vitor Euzebio  
- Paulo Sergio  
- Enzo Lucato  

---

## Objetivo

Desenvolver um sistema completo de previsão de churn de clientes utilizando Machine Learning, incluindo análise de dados, modelagem, avaliação e API.

---

## Problema de Negócio

Churn representa clientes que cancelam o serviço. O objetivo é identificar clientes com maior probabilidade de cancelamento.

---

## Estrutura do Projeto

```bash
tech-challenge-churn/

├── data/
│   └── raw/
│       └── telco_churn.csv
│
├── models/
│   ├── best_mlp.pth
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── docs/
│   ├── model_card.md
│   └── monitoring.md
│
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   └── models/
│
├── tests/
│
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Como executar

### Criar ambiente

```bash
python -m venv .venv
```

### Ativar

```bash
.venv\Scripts\activate
```

### Instalar dependências

```bash
pip install pandas numpy scikit-learn torch mlflow fastapi uvicorn joblib pytest ruff seaborn matplotlib
```

---

## Rodar testes

```bash
python -m pytest
```

---

## Treinar modelo

```bash
python -m src.models.train_model
```

---

## Baselines

```bash
python -m src.models.baseline_model
```

---

## MLflow

```bash
mlflow ui
```

Acesse:

```
http://127.0.0.1:5000
```

---

## API

```bash
python -m uvicorn src.api.main:app --reload
```

Acesse:

```
http://127.0.0.1:8000/docs
```

---

## Teste

### Churn

```json
{
  "gender": "Female",
  "SeniorCitizen": 1,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0,
  "TotalCharges": 95.0
}
```

---

### No Churn

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 60,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 50.0,
  "TotalCharges": 3000.0
}
```

---

## Conclusão

Projeto completo com pipeline de Machine Learning, MLflow e API funcional.
