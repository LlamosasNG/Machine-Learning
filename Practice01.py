import numpy as np

# 1.- Definir los datos de entrada 
X = np.array([1,2,3,4,5,6,7,8,9,10])
Yd = np.array([0.8,2.95,2.3,3.6,5.2,5.3,6.1,5.9,7.6,9])

# 2.- Definir parametros
a = 0   
b = 0
lr = 0.05
epocas = 4000
m = len(Yd)

Yobt = np.zeros(m)
for i in range(epocas):
    # 3.- Calculamos Yobt
    Yobt = a * X + b
    a =  ((m * np.sum(X*Yd)) - (np.sum(X) * np.sum(Yd)))/((m * np.sum(X**2)) - (np.sum(X)**2))
    b = (np.sum(Yd) - a * np.sum(X))/m
    ECM = (1 / (2 * m)) * np.sum(Yobt - Yd) **2
    print(ECM)
    print(Yobt)
 
""" print(a)
print(b)  """
