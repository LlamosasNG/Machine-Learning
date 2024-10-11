import numpy as np
import matplotlib.pyplot as plt

# 1.- Definir los datos de entrada 
X = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
Yd = np.array([5.1, 5.4, 5.7, 6, 6.9, 7.5, 8.8, 9.8, 10.5, 9.0])

# 2.- Definir parametros
a = 0.6
b = -1217.2
lr = 0.0000001
epocas = 4000  
m = len(Yd)

Yobt = np.zeros(m)
for i in range(epocas):
    # 3.- Calculamos Yobt
    Yobt = a * X + b
    a -= (lr / m) * np.sum((Yobt - Yd) * X)
    b -= (lr / m) * np.sum((Yobt - Yd))
    ECM = (1 / (2 * m)) * np.sum(Yobt - Yd) **2
    #print(ECM)
    
print(f"a = { a }")
print(f"b = { b }")

""" 
# 4.- Grafica de las funciones
plt.scatter(X, Yd, color='blue', label='Datos originales')  # Graficar puntos 
plt.plot(X, Yobt, color='red', label='Recta de regresión')  # Graficar línea

plt.xlabel('Año')  
plt.ylabel('MDD')  
plt.title('Regresión Lineal Vía Gradiente Descendente')  

plt.show()
 """