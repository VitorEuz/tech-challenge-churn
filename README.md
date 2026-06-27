# Tech Challenge - Churn Prediction

## Integrantes

* Vitor Euzebio
* Paulo Sergio
* Enzo Lucato

---

# Visão Geral

Este projeto apresenta uma solução completa de Machine Learning para previsão de churn de clientes em uma empresa de telecomunicações.

A aplicação contempla todo o ciclo de desenvolvimento de um modelo preditivo, desde o carregamento e tratamento dos dados até a disponibilização do modelo por meio de uma API REST.

O projeto também inclui:

* Análise exploratória dos dados (EDA);
* Modelos baseline para comparação;
* Rede Neural MLP implementada em PyTorch;
* Rastreamento de experimentos utilizando MLflow;
* Testes automatizados;
* API desenvolvida com FastAPI.

---

# Problema de Negócio

Churn representa clientes que cancelam um serviço.

Identificar clientes com maior probabilidade de cancelamento permite que a empresa realize ações preventivas de retenção, reduzindo perdas financeiras e aumentando o valor do cliente ao longo do tempo.

---

# Tecnologias Utilizadas

* Python 3.13
* Pandas
* NumPy
* Scikit-Learn
* PyTorch
* FastAPI
* MLflow
* Pytest
* Ruff

---

# Estrutura do Projeto

```text
tech-challenge-churn/

├── data/
│   └── raw/
│       └── telco_churn.csv
│
├── docs/
│   ├── model_card.md
│   └── monitoring.md
│
├── notebooks/
│   └── EDA.ipynb
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
├── requirements.txt
└── README.md
```

---

# Pré-requisitos

O projeto foi desenvolvido e testado utilizando:

* Python 3.13
* Git

Verifique sua versão do Python:

```bash
python --version
```

Resultado esperado:

```text
Python 3.13.x
```

---

# Instalação

## 1. Clonar o repositório

```bash
git clone https://github.com/VitorEuz/tech-challenge-churn.git
```

Entre na pasta:

```bash
cd tech-challenge-churn
```

---

## 2. Criar o ambiente virtual

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
```

---

## 3. Ativar o ambiente virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

# Validação da Instalação

## Verificar qualidade do código

```bash
python -m ruff check .
```

Resultado esperado:

```text
All checks passed!
```

---

## Executar os testes automatizados

Treinar o mode:

```bash
python -m src.models.train_model
```
Executar os testes:

```bash
python -m pytest
```

Resultado esperado:

```text
4 passed
```

---

# Treinar o Modelo

Execute:

```bash
python -m src.models.train_model
```

Durante o treinamento serão exibidas as métricas de treinamento e avaliação do modelo.

Ao final serão gerados automaticamente os artefatos necessários para inferência.

O treinamento também registra automaticamente:

* parâmetros;
* métricas;
* modelo treinado;
* artefatos;

no MLflow.

---

# Executar os Modelos Baseline

Para comparar o desempenho da MLP com modelos clássicos execute:

```bash
python -m src.models.baseline_model
```

Serão avaliados:

* Dummy Classifier
* Logistic Regression
* Random Forest

---

# MLflow

Inicie o servidor do MLflow:

```bash
mlflow ui
```

Abra no navegador:

```text
http://127.0.0.1:5000
```

No MLflow é possível visualizar:

* Experimentos
* Métricas
* Parâmetros
* Modelos registrados
* Comparação entre modelos

---

# Executar a API

Inicie a API:

```bash
python -m uvicorn src.api.main:app --reload
```

Abra a documentação automática:

```text
http://127.0.0.1:8000/docs
```

Endpoints disponíveis:

* GET `/health`
* POST `/predict`

---

# Teste da API

## Endpoint Health

Execute o endpoint:

```
GET /health
```

Resposta esperada:

```json
{
    "status": "ok"
}
```

---

## Cliente com Churn

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

Resposta esperada (valores aproximados):

```json
{
  "prediction": 1,
  "churn_probability": 0.94,
  "label": "Churn"
}
```

---

## Cliente sem Churn

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

Resposta esperada (valores aproximados):

```json
{
  "prediction": 0,
  "churn_probability": 0.10,
  "label": "No Churn"
}
```

---

# Considerações Finais

Este projeto entrega uma solução completa para previsão de churn utilizando técnicas modernas de Machine Learning.

A solução contempla:

* Pipeline completo de processamento de dados;
* Modelos baseline para comparação;
* Rede Neural MLP em PyTorch;
* Rastreamento de experimentos com MLflow;
* API REST para inferência;
* Testes automatizados;
* Código validado com Ruff.

Toda a aplicação foi organizada seguindo boas práticas de Engenharia de Machine Learning, buscando garantir reprodutibilidade, organização, escalabilidade e facilidade de manutenção.
