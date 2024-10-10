import numpy as np
import matplotlib.pyplot as plt
import time

# 1.- Definir los datos de entrada 
X = np.array([1,2,3,4,5,6,7,8,9,10])
Yd = np.array([0.8,2.95,2.3,3.6,5.2,5.3,6.1,5.9,7.6,9])

start_time = time.time()
# 2.- Calcular los coeficientes de la regresión lineal
m = len(Yd) 
a = ((m * np.sum(X * Yd)) - (np.sum(X) * np.sum(Yd))) / ((m * np.sum(X ** 2)) - (np.sum(X) ** 2))
b = (np.sum(Yd) - a * np.sum(X)) / m
print(f"a = {a}")
print(f"b = {b}")

# 3.- Calcular los valores estimados (Yobt)
Yobt = a * X + b
#print(Yobt)

ECM = (1 / (2*m)) * np.sum(Yobt - Yd) **2
#print(ECM)
end_time = time.time()
print(f"El tiempo de ejecución es de {end_time - start_time} segundos")

# 4.- Grafica de las funciones
plt.scatter(X, Yd, color='blue', label='Datos originales')  # Graficar puntos 
plt.plot(X, Yobt, color='red', label='Recta de regresión')  # Graficar línea

plt.xlabel('X')  
plt.ylabel('Yd')  
plt.title('Regresión Lineal Simple')  

plt.show()
