# Regresion lineal vía actualización de pesos
import numpy as np
import matplotlib.pyplot as plt
import time

# 1.- Definir los datos de entrada 
X = np.array([1,2,3,4,5,6,7,8,9,10])
Yd = np.array([0.8,2.95,2.3,3.6,5.2,5.3,6.1,5.9,7.6,9])

# 2.- Definir parametros
a = 0.7
b = 0.9
lr = 0.001
epocas = 4000
m = len(Yd)

Yobt = np.zeros(m)
start_time = time.time()
for i in range(epocas):
    # 3.- Calculamos Yobt
    Yobt = a * X + b
    a -= (lr / m) * np.sum((Yobt - Yd) * X)
    b -= (lr / m) * np.sum((Yobt - Yd))
    
    ECM = (1 / (2 * m)) * np.sum(Yobt - Yd) **2

print(f"a = { a }")
print(f"b = { b }")

end_time = time.time()
print(f"El tiempo de ejecución es de {end_time - start_time} segundos")

# 4.- Grafica de las funciones
plt.scatter(X, Yd, color='blue', label='Datos originales')  # Graficar puntos 
plt.plot(X, Yobt, color='red', label='Recta de regresión')  # Graficar línea

plt.xlabel('X')  
plt.ylabel('Yd')  
plt.title('Regresión Lineal Vía Actualización de Pesos')  

plt.show()
