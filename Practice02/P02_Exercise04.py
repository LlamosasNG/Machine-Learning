import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.01

# Definir la función sigmoide
def Sigmoide(z):
    return 1 / (1 + np.exp(-z))

def ValidateH(H):
    Predictions = np.zeros(len(H))  
    for i in range(len(H)):
        if H[i] > 0.5:
            Predictions[i] = 1
        else:
            Predictions[i] = 0
    return Predictions

# Datos de entrenamiento
quantity = int(input("Cantidad de datos que quieres ingresar en X e Yd \n"))
X = np.zeros(quantity)
Yd = np.zeros(quantity)

for i in range(quantity):
    X[i] = float(input(f"Valor de X[{i}]: "))
    Yd[i] = float(input(f"Valor de Yd[{i}]: "))

# Definiendo hiperparámetros
n = len(X[0])
theta = np.zeros(n)
theta0 = 0

# Hiperparámetros dentro del modelo
lr = 0.1
epocas = 1000
m = len(Yd)

J_hist = np.zeros(epocas)

for i in range(epocas):
    Z = np.dot(X, theta) 
    H = Sigmoide(Z)
    
    # Evaluar la función de costo
    J = -(1/m) * np.sum(Yd * np.log(H) + (1 - Yd) * np.log(1 - H))
    
    J_hist[i] = J
    
    theta0 = theta0 - lr * (1/m) * np.sum(H - Yd)
    theta = theta - lr * (1/m) * np.dot(X.T, (H - Yd))
    
    if abs(J_hist[i] - J_hist[i-1]) < epsilon:
        break

Yobt = ValidateH(H)

print(f"Theta0: {theta0}")
print(f"Theta: {theta}")

# Graficar los resultados
plt.scatter(X[:, 1], Yd, color='blue', label='Datos de entrenamiento')

J_hist_scaled = (J_hist - np.min(J_hist)) / (np.max(J_hist) - np.min(J_hist))
plt.plot(np.linspace(2, 21, epocas), J_hist_scaled, color='green', label='Función de costo J (escalada)')

plt.xlabel('X')
plt.ylabel('Yd / H')
plt.title('Regresión logística')
plt.legend()
plt.show()
