import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carrega o conjunto de dados
data = pd.read_csv("Parkinson disease.csv")  # Carrega o CSV "Parkinson disease.csv"

# Separa os dados em características (X) e variável alvo (y)
# "name" não é necessária e está em formato string, enquanto as outras características são valores flutuantes
X = data.drop({"status", "name"}, axis=1)  # Remove as colunas "status" e "name"
y = data["status"]  # Define a coluna "status" como a variável alvo

# Separa os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Divide os dados em 80% para treino (X_train, y_train) e 20% para teste (X_test, y_test)
# random_state=42 garante reproducibilidade dos resultados

# Normalização de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajusta e transforma os dados de treino
X_test_scaled = scaler.transform(X_test)  # Transforma os dados de teste usando o mesmo ajuste do treino

# Treinamento do modelo (Exemplo usando XGBoost)
model = XGBClassifier()
model.fit(X_train_scaled, y_train)  # Treina o modelo com os dados normalizados de treino

# Predição no conjunto de teste
y_pred = model.predict(X_test_scaled)  # Faz previsões usando o modelo treinado

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Acurácea:", accuracy)
print("Precisão:", precision)  # Corrigido: "Precisão" repetido, deve ser "Recall"
print("Recall:", recall)
print("F1-score:", f1)
