#  Model Card — Previsão de Churn (Telecom)

---

## Descrição do Modelo

Este projeto utiliza uma rede neural do tipo **MLP (Multi-Layer Perceptron)**, desenvolvida com PyTorch, para prever a probabilidade de churn (cancelamento) de clientes em uma operadora de telecomunicações.

O modelo foi treinado com dados tabulares contendo informações demográficas, contratuais e de uso dos clientes.

---

## Objetivo

Identificar clientes com alto risco de cancelamento, permitindo que a empresa tome ações preventivas de retenção e reduza perdas financeiras.

---

## Dados Utilizados

- Dataset: Telco Customer Churn  
- Tipo: dados estruturados (tabulares)  
- Volume: ~7.000 registros  
- Features:
  - Dados demográficos (gênero, dependentes, parceiro)
  - Informações contratuais (tipo de contrato, método de pagamento)
  - Serviços contratados (internet, streaming, suporte técnico)
  - Dados financeiros (MonthlyCharges, TotalCharges)
  - Tempo de relacionamento (tenure)

---

## Performance do Modelo

### 🔹 Baseline — Regressão Logística
- AUC-ROC: 0.84  
- F1 Score: 0.61  

### 🔹 Modelo Final — Rede Neural (MLP)
- AUC-ROC: 0.83  
- F1 Score: 0.57  
- Precision: 0.79  

Observação: O modelo MLP apresentou desempenho semelhante ao baseline, evidenciando que modelos mais complexos nem sempre superam modelos lineares em dados tabulares.

---

## Trade-off de Negócio

Foi realizada análise de trade-off entre falsos positivos e falsos negativos.

- **Falsos Negativos (FN)**: clientes que cancelam e não foram identificados → maior impacto financeiro  
- **Falsos Positivos (FP)**: clientes que não cancelariam, mas recebem incentivo → custo operacional  

Conclusão:  
O modelo pode ser ajustado para priorizar **recall**, reduzindo falsos negativos e maximizando retenção.

---

## ⚠️ Limitações

- Dataset desbalanceado (classe churn minoritária)
- Não considera comportamento temporal do cliente
- Sensível à qualidade e consistência dos dados
- Pode não generalizar bem para novos cenários sem re-treinamento

---

## Vieses

- Pode favorecer perfis mais frequentes no dataset
- Possível viés relacionado a tipo de contrato ou perfil de consumo
- Pode refletir padrões históricos da empresa, incluindo decisões passadas

---

## 🚨 Cenários de Falha

O modelo pode apresentar desempenho inferior em:

- Clientes novos (baixo tempo de contrato)
- Mudanças recentes de comportamento (não capturadas nos dados)
- Dados incompletos ou inconsistentes
- Perfis pouco representados no dataset

---

## Uso Recomendado

- Apoio a estratégias de retenção de clientes
- Identificação de clientes com alto risco de churn
- Segmentação para campanhas de marketing

⚠️ O modelo **não deve ser utilizado como única base para decisões automáticas**, devendo sempre ser acompanhado de análise humana.

---

## Atualização e Manutenção

- Recomenda-se re-treinamento periódico do modelo
- Monitoramento contínuo de performance em produção
- Avaliação de drift de dados e conceito

---

## Reprodutibilidade

- Seeds fixadas para experimentos
- Pipeline de pré-processamento definido
- Experimentos rastreados com MLflow
