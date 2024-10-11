import numpy as np
import matplotlib.pyplot as plt

def metricas_rend(Yobt, Yd):
    vp = 0
    vn = 0
    fn = 0
    fp = 0
    
    tamanioYd = len(Yd)
    tamanioYobt = len(Yobt)

    if tamanioYd == tamanioYobt:
        for i in range(tamanioYd):
            if Yd[i] == 1 and Yobt[i] == 1:
                vp = vp+1
            elif Yd[i] == 0 and Yobt[i] == 0:
                vn = vn+1
            elif Yd[i] == 1 and Yobt[i] == 0:
                fn = fn+1
            elif Yd[i] == 0 and Yobt[i] == 1:
                fp = fp+1
    
def asignar_clase(Yobt, Yd):
    clases = np.zeros(len(Yobt)) 
    for i in range(len(Yobt)):
        if Yobt[i] >= Yd[i]:
            clases[i] = 1  
        else:
            clases[i] = 0 
    return clases

# Datos iniciales
X = np.array([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5])
Yd = np.array([2,3,5,4,7,6,10,9,13,12])

a = 1.0  
b = -0.5  
epocas = 5000
lr = 0.01  
m = len(Yd)

Yobt = np.zeros(m)

for i in range(epocas):
    Yobt = a * X + b
    a -= (lr / m) * np.sum((Yobt - Yd) * X)
    b -= (lr / m) * np.sum(Yobt - Yd)

    ECM = (1 / (2 * m)) * np.sum((Yobt - Yd) ** 2)
    #print(ECM)
print(Yobt)

clases = asignar_clase(Yobt, Yd)

Xtest = np.array([11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5])
Ytest = np.array([14,11,13,16,19,20,24,21,26,22,28,30,32,34,36])

YobtTest = a * Xtest + b

clasesTest = asignar_clase(YobtTest, Ytest)

#Para presición y exactitud
metricas_rend(Yobt, Yd)
