# Detecção de Fraudes em Transações de Cartão de Crédito

Projeto de Machine Learning para detecção de transações fraudulentas em uma base real de transações de cartão de crédito.

O principal desafio do projeto é o **forte desbalanceamento entre as classes**: apenas 0,17% das transações são fraudes. Nesse cenário, a acurácia isoladamente pode transmitir uma visão enganosa do desempenho do modelo. Por isso, o projeto prioriza métricas como **Recall, Precision e F1-score para a classe de fraude**.

## Objetivo

Construir e comparar modelos capazes de identificar transações potencialmente fraudulentas, avaliando não apenas quantas previsões estão corretas, mas principalmente a capacidade de encontrar as fraudes sem gerar uma quantidade excessiva de falsos positivos.

Além da comparação entre modelos, o projeto também explora:

- técnicas de balanceamento;
- ajuste do limiar de decisão;
- importância das variáveis;
- explicabilidade com SHAP.

---

## Dataset

Foi utilizada a base de transações de cartão de crédito disponibilizada no desafio.

O dataset contém:

- **284.807 transações**
- **31 colunas**
- `Time`
- `Amount`
- `Class`
- variáveis `V1` a `V28`

As variáveis `V1` a `V28` são componentes transformados por PCA para preservar a privacidade das informações originais.

A coluna `Class` representa o alvo:

- `0` → transação normal
- `1` → transação fraudulenta

O dataset é carregado diretamente pela URL durante a execução do notebook e não é armazenado no repositório.

---

## Desbalanceamento

A distribuição encontrada foi:

| Classe | Quantidade | Percentual |
|---|---:|---:|
| Normal | 284.315 | 99,827% |
| Fraude | 492 | 0,173% |

Apenas **492 das 284.807 transações** são fraudulentas.

Isso significa que um modelo que classificasse praticamente todas as transações como normais poderia apresentar uma acurácia muito alta mesmo sendo pouco útil para detectar fraude.

Por esse motivo, o projeto utiliza principalmente:

- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- Average Precision;
- curva Precision-Recall.

O **Recall da classe fraude** recebe atenção especial porque representa a proporção das fraudes reais que foram identificadas pelo modelo.

---

## Preparação dos dados

A preparação foi realizada antes do treinamento dos modelos.

### Criação de variável

Foi criada a variável `LogAmount`, utilizando o logaritmo do valor da transação:

```python
df["LogAmount"] = np.log1p(df["Amount"])
```

Essa transformação ajuda a reduzir a influência da grande variação existente nos valores das transações.

### Separação dos dados

Os dados foram separados em:

- 80% para treinamento;
- 20% para teste.

Foi utilizado `stratify=y` para preservar a proporção de transações fraudulentas nos dois conjuntos.

Resultado:

| Conjunto | Registros | Fraude |
|---|---:|---:|
| Treino | 227.845 | 0,173% |
| Teste | 56.962 | 0,172% |

### Padronização

O `StandardScaler` foi utilizado nos modelos que trabalham com a escala das variáveis.

---

## Modelos utilizados

Foram treinados três modelos:

### Regressão Logística

Utilizada como modelo de baseline.

Foi configurado `class_weight="balanced"` para considerar o forte desbalanceamento entre as classes.

### Random Forest

Modelo baseado em múltiplas árvores de decisão.

Foi utilizado `class_weight="balanced_subsample"` para considerar o desbalanceamento durante o treinamento.

### XGBoost

Modelo baseado em árvores de decisão com boosting.

Foi utilizado `scale_pos_weight` calculado a partir da proporção entre as classes no conjunto de treinamento.

---

## Comparação dos modelos

Com o threshold padrão de **0,50**, os resultados foram:

| Modelo | Precision fraude | Recall fraude | F1 fraude | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,0603 | **0,9184** | 0,1131 | 0,9711 | 0,7113 |
| Random Forest | **0,9610** | 0,7551 | 0,8457 | 0,9471 | 0,8591 |
| XGBoost | 0,8632 | 0,8367 | **0,8497** | **0,9798** | **0,8765** |

Os resultados mostram diferentes comportamentos entre os modelos.

A Regressão Logística apresentou o maior recall entre os três modelos nesse threshold, identificando 91,84% das fraudes presentes no conjunto de teste. Porém, sua precisão foi de apenas 6,03%, indicando uma quantidade elevada de falsos positivos.

O Random Forest apresentou a maior precisão, com 96,10%, mas seu recall foi menor, de 75,51%.

O XGBoost apresentou um equilíbrio entre as métricas: recall de 83,67%, precisão de 86,32% e F1-score de 84,97%. Também apresentou o maior ROC-AUC e Average Precision entre os modelos avaliados.

---

## Por que a acurácia não é suficiente?

A acurácia pode ser elevada em problemas altamente desbalanceados mesmo quando o modelo não consegue identificar adequadamente a classe minoritária.

Neste projeto, existem 284.315 transações normais contra apenas 492 fraudes.

Por isso, a avaliação foi direcionada para o comportamento da classe fraude, especialmente o Recall, Precision e F1-score.

---

## Ajuste do limiar de decisão

O threshold padrão utilizado para classificação é 0,50:

```text
probabilidade >= 0,50 → fraude
probabilidade < 0,50 → normal
```

Entretanto, esse valor pode ser alterado.

Foram testados thresholds entre 0,05 e 0,95 para o XGBoost.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0,05 | 0,6069 | 0,8980 | 0,7243 |
| 0,10 | 0,7059 | 0,8571 | 0,7742 |
| 0,20 | 0,8077 | 0,8571 | 0,8317 |
| 0,30 | 0,8283 | 0,8367 | 0,8325 |
| 0,40 | 0,8542 | 0,8367 | 0,8454 |
| 0,50 | 0,8632 | 0,8367 | 0,8497 |
| 0,60 | 0,8723 | 0,8367 | 0,8542 |
| 0,70 | 0,8723 | 0,8367 | 0,8542 |
| 0,80 | 0,8901 | 0,8265 | 0,8571 |
| 0,90 | **0,9091** | 0,8163 | **0,8602** |
| 0,95 | 0,9186 | 0,8061 | 0,8587 |

Entre os thresholds testados, o maior F1-score foi obtido com **threshold de 0,90**:

- Precision: **90,91%**
- Recall: **81,63%**
- F1: **86,02%**

Nesse ponto, o modelo passa a exigir uma probabilidade maior para classificar uma transação como fraude. Como consequência, a precisão aumenta, enquanto o recall diminui em relação a thresholds mais baixos.

A escolha do threshold, portanto, representa um trade-off entre detectar mais fraudes e reduzir falsos positivos.

---

## Undersampling e Oversampling

Também foram testadas duas estratégias para lidar com o desbalanceamento.

### Undersampling

O undersampling reduz a quantidade de exemplos da classe majoritária.

Resultado:

| Técnica | Precision | Recall | F1 |
|---|---:|---:|---:|
| Undersampling | 0,0523 | 0,9184 | 0,0989 |

### Oversampling

O oversampling aumenta a representação da classe minoritária no conjunto de treinamento.

Resultado:

| Técnica | Precision | Recall | F1 |
|---|---:|---:|---:|
| Oversampling | 0,0603 | 0,9184 | 0,1131 |

Nos testes realizados, ambas as técnicas apresentaram recall elevado, mas precisão bastante baixa. Isso mostra que aumentar a capacidade de encontrar fraudes não significa necessariamente produzir um modelo com bom equilíbrio entre detecção e falsos positivos.

---

## Importância das variáveis

A importância das variáveis foi analisada inicialmente com o Random Forest.

As 15 variáveis com maior importância foram:

| Variável | Importância |
|---|---:|
| V14 | 0,1784 |
| V4 | 0,1173 |
| V10 | 0,1093 |
| V12 | 0,0931 |
| V17 | 0,0888 |
| V3 | 0,0624 |
| V11 | 0,0564 |
| V16 | 0,0510 |
| V2 | 0,0313 |
| V7 | 0,0258 |
| V21 | 0,0192 |
| V9 | 0,0188 |
| LogAmount | 0,0125 |
| V20 | 0,0124 |
| V18 | 0,0116 |

A variável `V14` apresentou a maior importância no Random Forest, seguida por `V4`, `V10`, `V12` e `V17`.

Como essas variáveis são componentes transformados por PCA, seus nomes não representam diretamente características compreensíveis do comportamento do cliente.

---

## Explicabilidade com SHAP

O SHAP foi utilizado para investigar como as variáveis contribuíram para as decisões do XGBoost.

A análise global permite observar quais variáveis tiveram maior influência nas previsões e em que direção seus valores contribuíram para aumentar ou reduzir a previsão de fraude.

Também foi feita uma análise individual de uma transação.

A transação selecionada apresentou:

- Probabilidade prevista de fraude: **97,63%**
- Classe real: **0 — normal**

Nesse caso, o modelo classificou a transação como fraude, embora ela fosse normal. Portanto, o exemplo escolhido representa um **falso positivo**.

O SHAP permite investigar quais variáveis contribuíram para que o modelo chegasse a essa previsão, tornando a decisão mais interpretável.

---

## O que foi acrescentado em relação ao caminho apresentado pela Expert

Além do fluxo principal apresentado no desafio, este projeto incorporou algumas análises adicionais:

- comparação entre três modelos;
- teste de undersampling;
- teste de oversampling;
- avaliação de diferentes thresholds;
- seleção de threshold pelo maior F1-score;
- comparação entre Precision, Recall e F1 em diferentes thresholds;
- análise de importância das variáveis;
- explicabilidade global com SHAP;
- explicação individual de uma transação;
- análise explícita de um falso positivo.

Essas etapas ajudaram a aprofundar a análise do impacto do desbalanceamento e do limiar de decisão no desempenho do detector de fraude.

---

## Principais aprendizados

O projeto mostrou que a avaliação de um modelo de fraude não deve ser baseada apenas em acurácia.

O forte desbalanceamento da base faz com que métricas como Recall, Precision e F1 sejam fundamentais para compreender o comportamento da classe minoritária.

Também foi possível observar que diferentes modelos apresentam diferentes compromissos entre encontrar fraudes e evitar falsos positivos.

O ajuste do threshold mostrou que a decisão final do modelo pode ser modificada de acordo com o objetivo da aplicação. No experimento realizado, aumentar o threshold do XGBoost para 0,90 aumentou a precisão e o F1 em relação ao threshold padrão de 0,50, ao mesmo tempo em que reduziu o recall.

Por fim, o uso do SHAP permitiu sair de uma análise baseada apenas em métricas e investigar quais variáveis contribuíram para determinadas previsões.

---

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib
- Google Colab

---

## Como executar

### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
```

### 2. Abra o notebook

Abra:

```text
deteccao_fraude_cartao.ipynb
```

no Google Colab ou em um ambiente Jupyter.

### 3. Execute as células

O dataset é baixado automaticamente pela URL:

```text
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Não é necessário baixar ou armazenar o dataset no repositório.

---

## Estrutura do projeto

```text
deteccao-fraude-cartao/
│
├── deteccao_fraude_cartao.ipynb
└── README.md
```

---

## Conclusão

A detecção de fraude em transações de cartão é um problema de classificação altamente desbalanceado, no qual a escolha das métricas e do limiar de decisão é tão importante quanto a escolha do algoritmo.

A comparação realizada mostrou comportamentos diferentes entre Regressão Logística, Random Forest e XGBoost. O XGBoost apresentou o maior ROC-AUC e Average Precision entre os modelos testados e, no ajuste de threshold, atingiu F1 de 0,8602 com threshold de 0,90.

O projeto também demonstrou que técnicas de balanceamento podem alterar significativamente a relação entre Recall e Precision e que ferramentas de explicabilidade como SHAP ajudam a compreender as decisões produzidas pelo modelo.