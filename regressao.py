# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generating a fictional dataset (100 samples, 1 independent variable, and 1 dependent variable)
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Independent variable
y = 2.5 * X.squeeze() + np.random.randn(100) * 2  # Dependent variable with some noise

# Dividing the dataset into training and testing sets (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating regression models
model_lr = LinearRegression()
model_dt = DecisionTreeRegressor(random_state=0)
model_knn = KNeighborsRegressor(n_neighbors=3)

# Training the models
model_lr.fit(X_train, y_train)
model_dt.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# Making predictions on the testing set
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_knn = model_knn.predict(X_test)

# Calculating performance metrics for each model
def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

# Evaluating the performance of each model
calculate_metrics(y_test, y_pred_lr, "Regressão Linear")
calculate_metrics(y_test, y_pred_dt, "Árvore de Decisão")
calculate_metrics(y_test, y_pred_knn, "KNN")

# Plotting the predicted values against the actual values for each model
plt.figure(figsize=(10, 6))

plt.plot(np.arange(len(y_test)), y_test, label="Valor Real", color="black", linewidth=2)
plt.plot(np.arange(len(y_test)), y_pred_lr, label="Regressão Linear", linestyle='--', marker='o')
plt.plot(np.arange(len(y_test)), y_pred_dt, label="Árvore de Decisão", linestyle='--', marker='s')
plt.plot(np.arange(len(y_test)), y_pred_knn, label="KNN", linestyle='--', marker='^')

plt.title("Previsões de diferentes modelos na porção de teste")
plt.xlabel("Índice da porção de teste")
plt.ylabel("Valores")
plt.legend()
plt.grid(True)
plt.show()