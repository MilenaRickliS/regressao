# Importando biblitecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


# Gerando um conjunto de dados mensal fictício para prever o lucro da empresa
np.random.seed(0)
months = np.arange(1, 13)  # 12 meses

# Dados mensais de marketing e investimento em P&D (em milhares)
marketing_investment = np.random.rand(12, 1) * 100  # Investimento em marketing (0 a 100k)
rnd_investment = np.random.rand(12, 1) * 200  # Investimento em P&D (0 a 200k)


# Adicionando sazonalidade aos lucros para simular flutuações mês a mês
seasonality = 10 * np.sin(2 * np.pi * months / 12) # Adicionando flutuações sazonais

# Combinando ambas as variáveis ​​independentes
X = np.hstack((marketing_investment, rnd_investment))

# Gerando a variável dependente (lucro) com sazonalidade e algum ruído
y = 5 * marketing_investment.squeeze() + 3 * rnd_investment.squeeze() + seasonality + np.random.randn(12) * 5  # Lucro em milhares

# Dividir o conjunto de dados em conjuntos de treinamento e teste (80% de treinamento e 20% de teste)
X_train, X_test, y_train, y_test, months_train, months_test = train_test_split(X, y, months, test_size=0.2, random_state=0)


# Gerando recursos polinomiais para regressão linear (grau 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# Atualizando o modelo de Regressão Linear com recursos polinomiais
model_lr_poly = LinearRegression()
model_lr_poly.fit(X_train_poly, y_train)

# Fazendo previsões no conjunto de testes com o modelo polinomial
y_pred_lr_poly = model_lr_poly.predict(X_test_poly)

# Recalcular métricas para o modelo de regressão polinomial
def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

calculate_metrics(y_test, y_pred_lr_poly, "Regressão Linear com Features Polinomiais")

# Criando modelos de regressão para outros modelos
model_lr = LinearRegression()
model_dt = DecisionTreeRegressor(random_state=0)
model_knn = KNeighborsRegressor(n_neighbors=3)

# Treinando os modelos
model_lr.fit(X_train, y_train)
model_dt.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# Fazendo previsões no conjunto de testes
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_knn = model_knn.predict(X_test)

# Avaliando o desempenho de cada modelo
calculate_metrics(y_test, y_pred_lr, "Regressão Linear")
calculate_metrics(y_test, y_pred_dt, "Árvore de Decisão")
calculate_metrics(y_test, y_pred_knn, "KNN")

# Plotando os valores previstos em relação aos valores reais de cada modelo
plt.figure(figsize=(10, 6))

# Traçando valores de lucro reais para o conjunto de dados completo
plt.plot(months, y, label="Valor Real", color="black", linewidth=2)

# Traçando as previsões de teste e os valores reais de teste
plt.scatter(months_test, y_test, color="black", label="Dados Reais de Teste", marker="x")
plt.plot(months_test, y_pred_lr_poly, label="Regressão Linear Polinomial", linestyle='--', marker='o')
plt.plot(months_test, y_pred_dt, label="Árvore de Decisão", linestyle='--', marker='s')
plt.plot(months_test, y_pred_knn, label="KNN", linestyle='--', marker='^')

plt.title("Previsões de Lucro Mensal da Empresa com Melhorias")
plt.xlabel("Meses")
plt.ylabel("Lucro (milhares de dólares)")
plt.xticks(months)
plt.legend()
plt.grid(True)
plt.show()
