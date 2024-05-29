## Classificação da Doença de Parkinson com XGBoost

**Objetivo:**

Este repositório contém um modelo de aprendizado de máquina para classificar pacientes com Doença de Parkinson utilizando o algoritmo XGBoost. O modelo foi treinado com um conjunto de dados disponível publicamente e pode ser usado para fazer previsões em novos dados.

**Instruções:**

1. **Clone o repositório:**

```bash
git clone https://github.com/<seu_usuario>/parkinson-classification-xgboost.git
```

2. **Instale as bibliotecas:**

```bash
pip install pandas scikit-learn xgboost
```

3. **Execute o script:**

```bash
python main.py
```

4. **Interpretação dos resultados:**

O script irá imprimir as seguintes métricas de avaliação do modelo:

* **Precisão:** Proporção de previsões corretas.
* **Recall:** Proporção de casos positivos reais corretamente identificados.
* **F1-score:** Média harmônica entre precisão e recall.

**Coleta de dados:**

O script também inclui uma função para coletar valores de características para um novo paciente. Para usá-la, siga estas etapas:

1. Execute o script.
2. Responda às perguntas com os valores das características do paciente.
3. Os valores coletados serão armazenados na lista `characteristics`.

**Observações:**

* Este modelo é apenas para fins de pesquisa e não deve ser usado para fins diagnósticos.
* É importante lembrar que a Doença de Parkinson é uma doença complexa e esse modelo pode não ser preciso para todos os casos.
* Se você tiver alguma dúvida ou precisar de ajuda, por favor, entre em contato comigo.

**Melhorias:**

* O modelo pode ser aprimorado com a utilização de um conjunto de dados maior e mais diverso.
* A seleção de hiperparâmetros do XGBoost pode ser otimizada para obter melhor desempenho.
* O modelo pode ser integrado a uma interface para facilitar o uso por usuários não técnicos.

**Licença:**

Este código é licenciado sob a licença MIT.
