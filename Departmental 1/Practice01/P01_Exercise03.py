import numpy as np
import matplotlib.pyplot as plt

def metricas_rend(Yobt, Yd):
  metricas = []
  vp = 0
  vn = 0
  fn = 0
  fp = 0

  tamanioYd = len(Yd)
  tamanioYobt = len(Yobt)

  if tamanioYd == tamanioYobt:
      for i in range(tamanioYd):
          if Yd[i] == 1 and Yobt[i] == 1:
              vp = vp + 1
          elif Yd[i] == 0 and Yobt[i] == 0:
              vn = vn + 1
          elif Yd[i] == 1 and Yobt[i] == 0:
              fn = fn + 1
          elif Yd[i] == 0 and Yobt[i] == 1:
              fp = fp + 1

  metricas = [vp, vn, fn, fp]

  return metricas

def asignar_clase(Yobt, Yd):
    clases = np.zeros(len(Yd))
    for i in range(len(Yobt)):
        if Yd[i] > Yobt[i]:
            clases[i] = 1
        else:
            clases[i] = 0
            
    return clases

# 1.- Definir los datos de entrada 
X = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
Yd = np.array([2, 3, 5, 4, 7, 6, 10, 9, 13, 12])
Ydclas = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1])

# 2.- Definir parametros
a = 1.0
b = -0.5
epocas = 5000
lr = 0.01
m = len(Yd)
Yobt = np.zeros(m)


for i in range(epocas):
    # 3.- Calculamos Yobt
    Yobt = a * X + b
    a -= (lr / m) * np.sum((Yobt - Yd) * X)
    b -= (lr / m) * np.sum(Yobt - Yd)
    ECM = (1 / (2 * m)) * np.sum((Yobt - Yd)**2)

print(f"Yobt = {Yobt}\n")

# 4.- Clasificacion de los dataset
clases = asignar_clase(Yobt, Yd)
print(f"Clasificacion Yobt: { clases }")

# 5.- Asignamos 15 datatest
Xtest = np.array([11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5])
Ytest = np.array([14, 11, 13, 16, 19, 20, 24, 21, 26, 22, 28, 30, 32, 34, 36])
YclasTest = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1])

YobtTest = a * Xtest + b

# 6.- Clasificacion de los datatest
clasesTest = asignar_clase(YobtTest, Ytest)
print(f"Clasificacion Datatest: { clasesTest }")

# 7.- Determinar Presicion y Exactitud
metricasYds = metricas_rend(clases, Ydclas)

# metricas = [vp,vn,fn,fp]
P = metricasYds[0] / (metricasYds[0] + metricasYds[3])
print(f"Precision = {P}")
Ex = (metricasYds[0] + metricasYds[1]) / (metricasYds[0] + metricasYds[1] + metricasYds[3] + metricasYds[2])
print(f"Exactitud = {Ex}")

# Determinar Recall
metricasYdt = metricas_rend(clasesTest, YclasTest)
metricasT = np.add(metricasYds, metricasYdt)
Re = metricasT[0] / (metricasT[0] + metricasT[2])

# 8.- Determinar F2 Score
F2 = (2 * P * Re) / (P + Re)
print(f"F2 Score: { F2 }")

# 9.- Graficar funciones
fig, (ax1, ax2) = plt.subplots(2, 1)

# Agregar títulos a cada subplot
ax1.set_title("Gráfica Dataset")
ax1.scatter(X, Yd)
ax1.plot(X, Yobt, color="red")

ax2.set_title("Gráfica Datatest")
ax2.scatter(Xtest, Ytest)
ax2.plot(Xtest, YobtTest, color="red")

# Mostrar las gráficas
plt.tight_layout()  # Asegura que los títulos no se sobrepongan
plt.show()
