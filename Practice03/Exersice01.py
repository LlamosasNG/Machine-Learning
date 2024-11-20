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
  
  return (x1, y1), (x2, y2), (x3, y3)

def normalization(X):
  X_mean = np.mean(X)
  X_std = np.std(X)
  X_stdized = (X - X_mean) / X_std

  return X_stdized
  
(x1, y1), (x2, y2), (x3, y3) = data_set()

# Definición de parámetros
lr = 0.01
epocas = 60000
Yd = y3
X = x3

# Solicitar el grado del polinomio
print("Ingrese el grado del polinomio entre 1 y 15: ")
n = int(input())
B = np.zeros(n)
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
  Yobt = np.dot(XX, B) # B0 es un escalar aquí

  # Determinar el error cuadrático medio
  ECM = (1/(2*m)) * np.sum((Yobt - Yd) ** 2)
  historial_ECM[i] = ECM

  # Descenso de gradiente
  B = B - (lr / m) * np.dot(XX.T, (Yobt - Yd))

# Graficar el ECM
plt.plot(historial_ECM)
plt.xlabel('Epocas')
plt.ylabel('ECM')
plt.title("Historial de ECM durante el entrenamiento")
plt.show()

# Visualización de la curva ajustada
plt.scatter(X, Yd, color="red", label="Datos originales")
x_plot = np.linspace(min(X), max(X), 100)
x_plot_norm = normalization(x_plot)

# Fase de comprobación (Construir la matriz de características)
XX_test = np.zeros((len(x_plot), n))
for j in range(n):
  XX_test[:, j] = x_plot_norm ** (j + 1)

y_plot = np.dot(XX_test, B)

plt.plot(x_plot, y_plot, color="blue", label="Ajuste Polinomial")
plt.title("Ajuste Polinomial")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True) 
plt.show()
