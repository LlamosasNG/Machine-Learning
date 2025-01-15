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
                # Actualización mínima para alejar w del margen
                w = w - lr * (2 * lamda * w)
            else:
                w = w - lr * (2 * lamda * w - np.dot(Y[i], x))
                b = b - lr * lamda * Y[i]

    # Identificar vectores de soporte considerando ambos márgenes
    tolerancia = 1
    vectores_soporte_indices = [
        i for i, x in enumerate(X)
        if abs(Y[i] * (np.dot(x, w) + b) - 1) <= tolerancia
    ]
    vectores_soporte = X[vectores_soporte_indices]
    clases_vectores_soporte = Y[vectores_soporte_indices]  # Extraer las clases de los vectores de soporte

    return w, b, vectores_soporte, vectores_soporte_indices, clases_vectores_soporte

# Predicción para un nuevo punto
def prediccion_MSV(X_test, w, b):
    return np.sign(np.dot(X_test, w) + b)

# Cargar los datos del archivo CSV
ruta_archivo = './train.csv'
datos = pd.read_csv(ruta_archivo)

# Extraer las columnas relevantes
Y = datos['Survived'].tolist()
vivos = [y for y in Y if y == 1][:30]
muertos = [y for y in Y if y == 0][:30]
Y_fixed = np.array(vivos + muertos)

# Filtrar los índices correspondientes
indices_vivos = [i for i, y in enumerate(Y) if y == 1][:30]
indices_muertos = [i for i, y in enumerate(Y) if y == 0][:30]
indices_filtrados = indices_vivos + indices_muertos

# Filtrar X_fixed con los índices seleccionados
SibSp = datos['SibSp'].tolist()
Age = datos['Age'].tolist()
Pclass = datos['Pclass'].tolist()

# Procesar valores faltantes o inconsistentes
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

# Crear la matriz de características X y filtrar por índices seleccionados
X = np.array(list(zip(Pclass, Age, SibSp)))
X_fixed = X[indices_filtrados]

# Entrenamiento con MSV
w, b, vectores_soporte, vectores_soporte_indices, clases_vectores_soporte = entrenamiento_MSV(X_fixed, Y_fixed)

nuevo_punto = []
nuevo_punto.append(int(input("Ingrese la clase de pasajero (Pclass, 1-3): ")))
nuevo_punto.append(float(input("Ingrese la edad del pasajero: ")))
nuevo_punto.append(int(input("Ingrese el número de hermanos/esposos (SibSp): ")))

nuevo_punto = np.array(nuevo_punto)
clase_predicha = prediccion_MSV(nuevo_punto, w, b)

# Mostrar en terminal
print(f"El nuevo punto pertenece a la clase: {'Sobrevivió' if clase_predicha == 1 else 'No sobrevivió'}")


# Imprimir los vectores de soporte y sus clases
print("Vectores de soporte:")
for i, vector in enumerate(vectores_soporte):
    print(f"Vector: {vector}, Clase: {clases_vectores_soporte[i]}")

# Graficar los datos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_fixed[:, 0], X_fixed[:, 1], X_fixed[:, 2], c=Y_fixed, cmap=plt.cm.Paired, label='Datos')

# Graficar vectores de soporte
ax.scatter(X_fixed[vectores_soporte_indices, 0],
           X_fixed[vectores_soporte_indices, 1],
           X_fixed[vectores_soporte_indices, 2],
           s=150, facecolors='yellow', edgecolors='red', linewidths=2, label='Vectores de Soporte')

# Graficar el nuevo punto
ax.scatter(nuevo_punto[0], nuevo_punto[1], nuevo_punto[2],
           color='green', marker='o', s=150, label='Nuevo punto')

ax.set_xlabel('Pclass')
ax.set_ylabel('Age')
ax.set_zlabel('SibSp')
plt.title('Máquinas de Soporte Vectorial (MSV)')
plt.legend()
plt.show()