# 📊 Monitoring — Previsão de Churn

---

## 🎯 Objetivo

Garantir que o modelo mantenha boa performance em produção ao longo do tempo.

---

## 📈 Métricas Monitoradas

- AUC-ROC
- F1 Score
- Taxa de churn prevista
- Taxa de erro da API

---

## 🚨 Alertas

Alertas serão acionados quando:

- AUC-ROC cair abaixo de 0.75
- F1 Score cair significativamente
- Aumento anormal de previsões de churn
- Falhas na API

---

## 🔄 Data Drift

Monitoramento de mudanças nos dados de entrada:

- Distribuição de tenure
- MonthlyCharges
- Tipo de contrato

---

## 🛠️ Plano de Ação

Em caso de problemas:

1. Verificar dados de entrada
2. Avaliar performance recente
3. Re-treinar modelo se necessário
4. Atualizar versão do modelo

---

## 🔁 Frequência

- Monitoramento diário
- Reavaliação mensal
