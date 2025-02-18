import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def entrenamiento_MSV(X, Y):
    numero_muestras, numero_caracteristicas = X.shape

    # Inicialización parámetros externos
    epocas = 1000
    lr = 0.01
    lamda = 1 / epocas

    # Inicialización de parámetros internos
    w = np.zeros(numero_caracteristicas) + 0.1
    b = 0.1

    for epoca in range(epocas):
        for i, x in enumerate(X):
            condicion_margen = Y[i] * (np.dot(x, w) + b) >= 1
            if condicion_margen:
                w = w - lr * (2 * lamda * w)
            else:
                w = w - lr * (2 * lamda * w - np.dot(Y[i], x))
                b = b - lr * lamda * Y[i]

    tolerancia = 0.9
    vectores_soporte_indices = [
        i for i, x in enumerate(X)
        if abs(Y[i] * (np.dot(x, w) + b) - 1) <= tolerancia
    ]
    vectores_soporte = X[vectores_soporte_indices]

    return w, b, vectores_soporte, vectores_soporte_indices

def prediccion_MSV(X_test, w, b):
    return np.sign(np.dot(X_test, w) + b)

# Cargar los datos
ruta_archivo = './train.csv'
datos = pd.read_csv(ruta_archivo)

Y = datos['Survived'].tolist()
vivos = [y for y in Y if y == 1][:30]
muertos = [y for y in Y if y == 0][:30]
Y_fixed = np.array(vivos + muertos)

indices_vivos = [i for i, y in enumerate(Y) if y == 1][:30]
indices_muertos = [i for i, y in enumerate(Y) if y == 0][:30]
indices_filtrados = indices_vivos + indices_muertos

SibSp = datos['SibSp'].tolist()
Age = datos['Age'].tolist()
Pclass = datos['Pclass'].tolist()

moda_sibsp = max(set(SibSp), key=SibSp.count)
SibSp = [moda_sibsp if pd.isna(s) else s for s in SibSp]
SibSp = [moda_sibsp if s < 0 or not isinstance(s, int) else s for s in SibSp]
SibSp = [int(s) for s in SibSp]

Age = [np.nan if pd.isna(a) else a for a in Age]
promedio_edad = np.nanmean(Age)
Age = [promedio_edad if np.isnan(a) else a for a in Age]

moda_clase = max(set(Pclass), key=Pclass.count)
Pclass = [moda_clase if pd.isna(c) else c for c in Pclass]
Pclass = [int(c) for c in Pclass]

X = np.array(list(zip(Pclass, Age, SibSp)))
X_fixed = X[indices_filtrados]

w, b, vectores_soporte, vectores_soporte_indices = entrenamiento_MSV(X_fixed, Y_fixed)

# Solicitar un nuevo punto al usuario
nuevo_punto = []
nuevo_punto.append(int(input("Ingrese la clase de pasajero (Pclass, 1-3): ")))
nuevo_punto.append(float(input("Ingrese la edad del pasajero: ")))
nuevo_punto.append(int(input("Ingrese el número de hermanos/esposos (SibSp): ")))

nuevo_punto = np.array(nuevo_punto)
clase_predicha = prediccion_MSV(nuevo_punto, w, b)

# Mostrar en terminal
resultado = "Sobrevivió" if clase_predicha == 1 else "No sobrevivió"
print(f"El nuevo punto pertenece a la clase: {resultado}")

# Gráfica tridimensional
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar los datos
ax.scatter(X_fixed[:, 0], X_fixed[:, 1], X_fixed[:, 2], c=Y_fixed, cmap=plt.cm.Paired, label='Datos')

# Graficar vectores de soporte
ax.scatter(X_fixed[vectores_soporte_indices, 0],
           X_fixed[vectores_soporte_indices, 1],
           X_fixed[vectores_soporte_indices, 2],
           s=150, facecolors='yellow', edgecolors='red', linewidths=2, label='Vectores de Soporte')

# Graficar el nuevo punto
ax.scatter(nuevo_punto[0], nuevo_punto[1], nuevo_punto[2],
           color='green', marker='o', s=150, label='Nuevo punto')

# Generar un rango para los planos
x = np.linspace(X_fixed[:, 0].min(), X_fixed[:, 0].max(), 10)
y = np.linspace(X_fixed[:, 1].min(), X_fixed[:, 1].max(), 10)
x, y = np.meshgrid(x, y)

# Hiperplano: z = -(w[0]*x + w[1]*y + b) / w[2]
z = -(w[0] * x + w[1] * y + b) / w[2]

# Márgenes: z = -(w[0]*x + w[1]*y + b ± 1) / w[2]
z_margen_superior = -(w[0] * x + w[1] * y + (b + 1)) / w[2]
z_margen_inferior = -(w[0] * x + w[1] * y + (b - 1)) / w[2]

# Graficar el hiperplano y los márgenes
ax.plot_surface(x, y, z, alpha=0.3, color='blue', label='Hiperplano')
ax.plot_surface(x, y, z_margen_superior, alpha=0.2, color='green', label='Margen Superior')
ax.plot_surface(x, y, z_margen_inferior, alpha=0.2, color='red', label='Margen Inferior')

ax.set_xlabel('Pclass')
ax.set_ylabel('Age')
ax.set_zlabel('SibSp')
plt.title('Máquinas de Soporte Vectorial (MSV)')
plt.legend()
plt.show()
