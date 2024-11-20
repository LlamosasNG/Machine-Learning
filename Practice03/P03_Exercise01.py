import numpy as np
import matplotlib.pyplot as plt

def data_set():
  # Set 1 Data CUADRÁTICA
  x1 = np.linspace(-10, 10, 20)
  y1 = 0.7 * x1**2 + 1.7 * x1 + 2 * np.random.normal(0, 10, x1.shape)

  # Set 2 Data CÚBICA
  x2 = np.linspace(-10, 10, 20)
  y2 = 0.3 * x2**3 - 0.78 * x2**2 + 1.7 * x2 + 4 * np.random.normal(0, 15, x2.shape)

  # Set 3 Data POLINOMIAL
  x3 = np.linspace(-30, 30, 20)
  y3 = 0.0025 * x3**4 - 0.003 * x3**3 + 0.008 * x3**2 + 0.007 * x3 + 0.08 * np.random.normal(-5, 5, x3.shape)
  
  # Set 4 Data POLINOMIAL
  x4 = np.linspace(-25, 25, 20)
  y4 = (0.002 * x4**8 - 0.005 * x4**7 + 0.01 * x4**6 - 0.02 * x4**5 + 0.03 * x4**4 - 0.04 * x4**3  + 0.05 * x4**2 - 0.06 * x4 + 2 + 10 * np.sin(x4) + 5 * np.cos(0.5 * x4) + np.random.normal(0, 20, x4.shape))
  
  return (x1, y1), (x2, y2), (x3, y3), (x4, y4)

def normalization(X):
  X_max_abs = np.max(np.abs(X))
  return X / X_max_abs

(x1, y1), (x2, y2), (x3, y3), (x4, y4) = data_set()

# Definición de parámetros
lr = 0.01
epocas = 60000
Yd = y3
X = x3

# Solicitar el grado del polinomio
print("Ingrese el grado del polinomio entre 1 y 15: ")
n = int(input())
B = np.zeros(n)
B0 = 0
m = len(Yd)

# Normalización de los vectores
X_norm = normalization(X)

# Inicializar historial de ECM
historial_ECM = np.zeros(epocas)

for i in range(epocas):
  XX = np.zeros((m, n))
  for j in range(n):
    XX[:, j] = X_norm ** (j + 1)

  # Determinar Yobt
  Yobt = np.dot(XX, B) + B0  # B0 es un escalar aquí

  # Determinar el error cuadrático medio
  ECM = (1/(2*m)) * np.sum((Yobt - Yd) ** 2)
  historial_ECM[i] = ECM

  # Descenso de gradiente
  B = B - (lr / m) * np.dot(XX.T, (Yobt - Yd))
  B0 = B0 - (lr / m) * np.sum(Yobt - Yd)

# Graficar el ECM
plt.plot(historial_ECM)
plt.xlabel('Epocas')
plt.ylabel('ECM')
plt.title("Historial de ECM durante el entrenamiento")
plt.show()

# Visualización de la curva ajustada
x_plot = np.linspace(min(X), max(X), 100)
x_plot_norm = normalization(x_plot)

# Fase de comprobación (Construir la matriz de características)
XX_test = np.zeros((len(x_plot), n))
for j in range(n):
  XX_test[:, j] = x_plot_norm ** (j + 1)

y_plot = np.dot(XX_test, B) + B0

# Clasificación en base a Yobt
labels = (Yd >= Yobt).astype(int)  # Clase 1: Por encima de Yobt, Clase 0: Por debajo

# Visualización de la clasificación
plt.figure(figsize=(10, 6))
plt.scatter(X[labels == 0], Yd[labels == 0], marker='x', color="red", label="Clase 0")
plt.scatter(X[labels == 1], Yd[labels == 1], marker='o', color="blue", label="Clase 1")
plt.plot(x_plot, y_plot, color="black", linestyle="--", label="Ajuste Polinomial")
plt.title("Clasificación basada en Yobt")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()