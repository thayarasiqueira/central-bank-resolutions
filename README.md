# Projeto de Análise de Complexidade Linguística

## Objetivo

Este projeto visa analisar o impacto da complexidade linguística nas resoluções do Banco Central do Brasil sobre a qualidade da classificação automática desses documentos.

## Estrutura do Projeto

- `data_mining/`: Contém scripts para mineração e análise de dados.
  - `main.py`: Script principal para execução do pipeline de análise.
  - `complexity_analysis.py`: Calcula métricas de complexidade linguística.
  - `categorization_model.py`: Treina e avalia modelos de classificação.
  - `validation.py`: Facilita a validação manual de amostras.
  - `statistical_analysis.py`: Realiza análises estatísticas.

- `data_collection/`: Scripts para coleta de dados.

## Como Executar

1. **Coleta de Dados**: Execute `data_collection/main.py` para coletar resoluções.
2. **Análise de Dados**: Execute `data_mining/main.py` para processar e analisar os dados.
3. **Validação de Amostras**: Use `data_mining/validation.py` para validar manualmente uma amostra dos dados.
4. **Análise Estatística**: Use `data_mining/statistical_analysis.py` para realizar análises estatísticas.

## Requisitos

- Python 3.8+
- Bibliotecas: numpy, pandas, scipy, matplotlib, seaborn, sklearn, nltk, spacy, gensim, transformers, tensorflow, keras

