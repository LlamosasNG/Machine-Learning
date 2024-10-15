import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.000001

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
X = np.array([[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],[1,18],[1,19],[1,20],[1,21]])
Yd = np.array([0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0])
Yobt = np.zeros(len(Yd))

# Definiendo hiperparámetros
n = len(X[0])
theta = np.zeros(n)  
theta0 = 0

# Hiperparámetros dentro del modelo
lr = 0.04
epocas = 1000
m = len(Yd)

# Array para almacenar los valores de J
J_hist = np.zeros(epocas)

for i in range(epocas):
    Z = theta0 + np.dot(X, theta)
    H = Sigmoide(Z)
    
    # Evaluar la función de costo
    J = -(1/m) * np.sum(Yd * np.log(H) + (1 - Yd) * np.log(1 - H))
    
    J_hist[i] = J

    theta0 = theta0 - lr * (1/m) * np.sum(H - Yd)
    theta = theta - lr * (1/m) * np.dot(X.T, (H - Yd))
    
    print(J_hist[i] - J_hist[i-1])
    if i > 0 and abs(J_hist[i] - J_hist[i-1]) < epsilon:
        print(f"Termina en {i} epocas")
        break

Yobt = ValidateH(H)

print(f"Theta0: {theta0}")
print(f"Theta: {theta}")

plt.scatter(X[:, 1], Yd, color='blue', label='Datos de entrenamiento')

J_hist_scaled = (J_hist - np.min(J_hist)) / (np.max(J_hist) - np.min(J_hist))
#plt.plot(np.linspace(2, 21, epocas), J_hist_scaled, color='green', label='Función de costo J (escalada)')
plt.plot(X[:, 1], H, color='red', label='Regresión logística')


plt.xlabel('X')
plt.ylabel('Yd / H')
plt.title('Regresión logística')
plt.legend()
plt.show()
