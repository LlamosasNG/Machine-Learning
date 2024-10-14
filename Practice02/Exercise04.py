import numpy as np
import matplotlib.pyplot as plt

# Definir la función sigmoide
def Sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Corregir la función ValidateH para trabajar con arrays
def ValidateH(H):
    Predictions = np.zeros(len(H))  # El tamaño debe ser dinámico
    for i in range(len(H)):
        if H[i] > 0.5:
            Predictions[i] = 1
        else:
            Predictions[i] = 0
    return Predictions

# Datos de entrenamiento
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
Yd = np.array([0, 0, 1, 1])
Yobt = np.zeros(len(Yd))

# Definiendo hiperparámetros 
theta = np.zeros(X.shape[1])

# Hiperparámetros dentro del modelo
lr = 0.1
epocas = 1000
m = len(Yd)

# Array para almacenar los valores de J
J_hist = np.zeros(epocas)

for i in range(epocas):
    # Realizar el producto punto de cada muestra con el vector theta
    Z = np.dot(X, theta)
    H = Sigmoide(Z)
    
    # Evaluar la función de costo
    J = -(1/m) * np.sum(Yd * np.log(H) + (1 - Yd) * np.log(1 - H))
    
    # Guardar el valor de J en cada época
    J_hist[i] = J

    # Determinar theta
    theta = theta - lr * (1/m) * np.dot(X.T, (H - Yd))

# Predicciones binarias
Yobt = ValidateH(H)

# Graficar los puntos de X y Yd junto con la evolución de la función de costo en un solo gráfico
plt.scatter(X[:, 1], Yd, color='blue', label='Datos de entrenamiento')
# plt.plot(X[:, 1], H, color='red', label='Regresión logística')

# Escalar la función de costo para que sea visible junto con los puntos (ya que su magnitud es diferente)
J_hist_scaled = (J_hist - np.min(J_hist)) / (np.max(J_hist) - np.min(J_hist))  # Escalar entre 0 y 1
plt.plot(np.linspace(2, 5, epocas), J_hist_scaled, color='green', label='Función de costo J (escalada)')

plt.xlabel('X')
plt.ylabel('Yd / H')
plt.title('Datos de entrenamiento, Regresión logística y Función de Costo')
plt.legend()
plt.show()
