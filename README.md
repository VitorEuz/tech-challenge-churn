# Tech Challenge - Churn Prediction

## Integrantes

- Vitor Euzebio  
- Paulo Sergio  
- Enzo Lucato  

---

## Visão Geral

Este projeto tem como objetivo desenvolver um pipeline completo de Machine Learning para previsão de churn de clientes em uma empresa de telecomunicações.

A solução contempla desde a análise exploratória dos dados até a disponibilização de uma API para consumo do modelo, incluindo rastreamento de experimentos com MLflow e validação por testes automatizados.

---

## Problema de Negócio

Churn representa clientes que cancelam um serviço. A antecipação desse comportamento permite ações estratégicas de retenção, reduzindo perdas financeiras e aumentando o valor do cliente ao longo do tempo.

O modelo desenvolvido busca identificar clientes com maior probabilidade de cancelamento com base em suas características.

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
pip install -r requirements.txt
```

---

## Treinar modelo

```bash
python -m src.models.train_model
```

Isso irá gerar os artefatos do modelo na pasta `models/`.

---

## Rodar testes

```bash
python -m pytest
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

## Teste da API

### Cliente com churn

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

### Cliente sem churn

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

## Considerações Finais

O projeto entrega uma solução completa de Machine Learning aplicada a um problema real de negócio, incluindo:

- Pipeline de dados estruturado  
- Modelagem com rede neural (MLP)  
- Comparação com modelos baseline  
- Monitoramento de experimentos com MLflow  
- API para consumo do modelo  
- Testes automatizados garantindo confiabilidade  

A arquitetura foi construída seguindo boas práticas de engenharia de Machine Learning, visando reprodutibilidade, organização e facilidade de manutenção.

---
