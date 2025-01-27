import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados fictícios
data = {
    'size': [50, 60, 70, 80, 90],         # Tamanho da casa (m²)
    'bedrooms': [1, 2, 2, 3, 3],         # Número de quartos
    'price': [150000, 180000, 210000, 240000, 270000]  # Preço da casa
}
df = pd.DataFrame(data)

# Variáveis independentes (X) e dependente (y)
X = df[['size', 'bedrooms']]
y = df['price']

# Dividir em treino e teste
# Explicação do metodo dentro do readme.md
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio (MSE):", mse)

# Exibir os coeficientes do modelo
print("Intercepto (β₀):", model.intercept_)
print("Coeficientes (β₁, β₂):", model.coef_)