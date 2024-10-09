import numpy as np

# Definir la función sigmoide
def Sigmoide(z):
    return 1/(1 + np.exp(-z))
    # z = theta * X

# Datos de entrenamiento
X = np.array([[1,2], [1,3], [1,4], [1,5],[1,6],[1,7],[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],[1,18],[1,19],[1,20],[1,21]])
Yd = np.array([0, 0, 1, 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
Yobt = np.zeros(len(Yd))

# Definiendo hiperparámetros 
theta1 = np.array([0, 0])
theta2 = np.array([0, 0])

# Hiperpárametros dentro del modelo
lr = 0.1
epocas = 1000
m = len(Yd)

for i in range(epocas):
    # Realizar el producto punto de cada una de mis muestras con el vector theta
    Z = np.dot(X, theta1)+theta2
    H = Sigmoide(Z)
      
    # Evaluar la función de costo
    J = -(1/m) * np.sum(Yd * np.log(H) + (1-Yd) * np.log(1-H))    

    #Calcular el gradiente y actualizar theta
    theta1 = theta1 - lr * (1/m) * np.dot(X.T,(H-Yd))
    theta2 = theta2 - lr * (1 / m) * np.sum((H - Yd))
print(H)
print(X)
print(theta1)
print(theta2)
    
    