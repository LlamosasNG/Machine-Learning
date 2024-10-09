import numpy as np

# Definir la función sigmoide
def Sigmoide(Z):
    return 1/(1 + np.exp(-Z))
    # z = theta * X

# Datos de entrenamiento
X = np.array([[1,2], [1,3], [1,4], [1,5]])
Yd = np.array([0, 0, 1, 1])
Yobt = np.zeros(len(Yd))

# Definiendo hiperparámetros 
theta = np.array([.5, 1])

# Hiperpárametrosdentro del modelo
lr = 0.01
epocas = 200
m = len(Yd)

# for i in range(epocas):
# Realizar el producto punto de cada una de mis muestras con el vector theta
Z = np.dot(X, theta)
print(Z)
H = Sigmoide(Z)

""" print(H)
print(np.log(H))
print(np.log(1 - H)) """

for i in range(len(H)):
    if (H[i] >= 0.5):
        Yobt[i] = 1
    else:
        Yobt[i] = 0
        
# Evaluar la función de costo
J = -(1 / m) * np.sum(Yd * np.log(H) + (1 - Yd) * np.log(1 - H))
#print(J)
""" theta = theta - lr * (1/m) * np.dot((Yd - H), X)
print(np.dot((Yd - H), X)) """
