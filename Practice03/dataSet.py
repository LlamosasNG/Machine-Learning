import numpy as np
import matplotlib.pyplot as plt

# Generar puntos distribuidos alrededor de una onda senoidal para clasificación
np.random.seed(42)

# Generar valores de x uniformemente distribuidos
x = np.linspace(-10, 10, 500)

# Generar valores de y basados en una función seno con ruido
y = 5 * np.sin(x) + np.random.normal(0, 1, x.shape)

# Clasificar puntos según si están por encima o por debajo de la onda senoidal
labels = (y >= 5 * np.sin(x)).astype(int)

# Visualizar los puntos clasificados
plt.figure(figsize=(10, 6))
plt.scatter(x[labels == 0], y[labels == 0], marker='o', color='red', label='Clase 0')
plt.scatter(x[labels == 1], y[labels == 1], marker='x', color='blue', label='Clase 1')
plt.title("Clasificación de puntos basados en una onda senoidal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

