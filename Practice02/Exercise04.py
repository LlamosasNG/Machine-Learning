import numpy as np

# Definir la función sigmoide
def Sigmoide(z):
    return 1/(1 + np.exp(-z))
    # z = theta * X

# Datos de entrenamiento
X = np.array([[1,2], [1,3], [1,4], [1,5]])
Yd = np.array([0, 0, 1, 1])
Yobt = np.zeros(len(Yd))

# Definiendo hiperparámetros 
theta = np.array([0, 0])

# Hiperpárametros dentro del modelo
lr = 0.1
epocas = 1000
m = len(Yd)

for i in range(epocas):
    # Realizar el producto punto de cada una de mis muestras con el vector theta
    Z = np.dot(X, theta)
    H = Sigmoide(Z)
      
    # Evaluar la función de costo
    J = -(1/m) * np.sum(Yd * np.log(H) + (1-Yd) * np.log(1-H))    

    #Calcular el gradiente y actualizar theta
    theta = theta - lr * (1/m) * np.dot(X.T,(H-Yd))
print(H)
print(X)
print(theta)
    
    