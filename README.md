# BreakPredict - Preditor de Falhas de Máquinas

Notebook de Machine Learning que prevê se uma máquina industrial vai falhar a partir das leituras dos seus sensores (temperatura, rotação, torque e desgaste da ferramenta), comparando dois modelos de classificação: **KNN** e **Árvore de Decisão**.

Projeto Avaliativo do Módulo 1 do curso *Desenvolvimento de IA para Análise Preditiva* (SCTEC).

## Problema

Uma falha inesperada para a linha de produção e custa reparo, peças e tempo parado. Se a falha puder ser prevista a partir dos sensores, a manutenção pode ser feita antes da quebra.

A base tem 10.000 registros e só **3,4% deles são falhas** (339). Esse desbalanceamento é a principal dificuldade do projeto: um modelo que dissesse sempre "a máquina está funcionando" acertaria 96,6% das vezes sem detectar nenhuma falha. Por isso a acurácia sozinha não serve para escolher o modelo, e a avaliação também usa recall, precision e matriz de confusão.

## Técnicas e tecnologias

O notebook `manutencao_preditiva.ipynb` segue 7 fases, cada uma justificada em células de texto:

| Fase | O que foi feito |
|---|---|
| 1. EDA | Dimensões, tipos, estatísticas, histogramas, desbalanceamento do alvo e mapa de calor de correlação |
| 2. Data Prep | Checagem de duplicados (inclusive ignorando os identificadores), imputação dos nulos pela mediana e análise de outliers com boxplot e IQR (outliers mantidos, por concentrarem falhas) |
| 3. Feature Engineering | Nova variável `potencia_kw = rotação × torque / 9549` (potência mecânica em kW) |
| 4. Divisão e balanceamento | Remoção de identificadores e das colunas de motivo de falha (vazamento), codificação ordinal do `tipo` (L/M/H), split 80/20 estratificado e SMOTE e Random Under Sampling **aplicados só no treino** |
| 5. Escalonamento | `StandardScaler` nas variáveis contínuas para o KNN (`fit` no treino, `transform` no teste); a Árvore usa os dados sem escala |
| 6. Hiperparâmetros | KNN com K ímpares de 3 a 13 e Árvore com várias profundidades, comparando acurácia de treino e teste para identificar overfitting |
| 7. Veredito | Comparação dos finalistas com acurácia, `classification_report` e matrizes de confusão |

**Tecnologias:** Python 3.13, pandas, Matplotlib, Seaborn, scikit-learn, imbalanced-learn e Jupyter Notebook.

## Resultado

| Modelo (base de teste) | Acurácia | Falhas detectadas (recall) | Alarmes falsos |
|---|---|---|---|
| KNN (K = 3, SMOTE, sem `potencia_kw`) | 92,4% | 44 de 68 (65%) | 129 |
| **Árvore de Decisão (SMOTE, sem limite de profundidade)** | **95,0%** | **48 de 68 (71%)** | **80** |

A Árvore de Decisão foi melhor em todos os indicadores e é o modelo recomendado.

## Como executar

Pré-requisitos: Python 3.13 e Git.

```bash
git clone https://github.com/filipester/manutencao_preditiva.git
cd manutencao_preditiva
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No Linux/macOS, use `python3.13 -m venv .venv` e `source .venv/bin/activate`.

Depois, abra o `manutencao_preditiva.ipynb`:

- **VS Code:** com a extensão Jupyter instalada, selecione o kernel da `.venv` e use *Run All*.
- **Jupyter Notebook:** instale com `pip install notebook` e rode `jupyter notebook`.

O dataset já está em `data/manutencao_preditiva.csv` e é lido por caminho relativo, então não é preciso configurar nada.

## Melhorias possíveis

- **Imputação depois do split:** a mediana foi calculada na base inteira, antes da divisão, seguindo a ordem do enunciado. O correto seria calcular só no treino, para evitar um vazamento leve de informação do teste.
- **Pipeline do imbalanced-learn:** o SMOTE rodou antes do escalonamento, então a escolha dos vizinhos sintéticos foi dominada pela rotação, que tem a maior escala. Um `Pipeline` (escalonar → SMOTE → modelo) resolve isso.
- **SMOTENC** para tratar o `tipo` como categórico na geração das falhas sintéticas.
- **Overfitting da Árvore:** a finalista tem 99,5% de acurácia no treino. Vale testar profundidades intermediárias e poda.
- **Validação cruzada** para escolher os hiperparâmetros com resultados mais estáveis do que um único split.
- **Ajuste do limiar de decisão** para priorizar o recall, já que uma falha não detectada custa mais do que uma inspeção desnecessária.

## Vídeo

Apresentação do projeto: [Link do vídeo](https://drive.google.com/file/d/1Kl58adQ9kRjPkv9Ko6hICyjDx87V4lfa/view?usp=drive_link)
