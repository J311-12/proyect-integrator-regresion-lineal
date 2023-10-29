import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargando los datos
df = pd.read_csv("datos_limpios.csv")

# Eliminando las columnas DEATH_EVENT, age y categoria_edad
df = df.drop(columns=["is_dead", "age", "edad_categoria"])

# Convirtiendo el DataFrame a un NumPy array
X = df.values

# Obteniendo el vector objetivo
y = df["age"].values

# Ajustando la regresión lineal
reg = LinearRegression().fit(X, y)

# Predeciendo las edades
y_pred = reg.predict(X)

# Calculando el error cuadrático medio
mse = np.mean((y_pred - y)**2)

# Imprimiendo el error cuadrático medio
print("Error cuadrático medio:", mse)
