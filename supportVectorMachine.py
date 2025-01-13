import numpy as np
import matplotlib.pyplot as plt

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
    tolerancia = .0005 
    vectores_soporte_indices = [
        i for i, x in enumerate(X)
        if abs(Y[i] * (np.dot(x, w) + b) - 1) <= tolerancia  # Cercanía a los márgenes
    ]
    vectores_soporte = X[vectores_soporte_indices]

    # Cálculo del margen del hiperplano
    M = 1 / np.linalg.norm(w)

    return w, b, vectores_soporte, vectores_soporte_indices

# Fase de operación
def prediccion_MSV(X_test, w, b):
    return np.sign(np.dot(X_test, w) + b)

# Datos fijos de ejemplo
X_fixed = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                    [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
                    [-1, -2], [-2, -3], [-3, -4], [-4, -5], [-5, -6],
                    [-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5]])

Y_fixed = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

w, b, vectores_soporte, vectores_soporte_indices = entrenamiento_MSV(X_fixed, Y_fixed)

# Graficamos los datos, el margen, los vectores de soporte y el hiperplano
plt.scatter(X_fixed[:, 0], X_fixed[:, 1], c=Y_fixed, cmap=plt.cm.Paired, label='Datos')
plt.scatter(X_fixed[vectores_soporte_indices, 0], X_fixed[vectores_soporte_indices, 1],
            s=150, facecolors='yellow', edgecolors='red', linewidths=2, marker='o', label='Vectores de Soporte')
plt.xlabel('Características 1')
plt.ylabel('Características 2')

# Graficar el margen
xx, yy = np.meshgrid(np.arange(X_fixed[:, 0].min() - 1, X_fixed[:, 0].max() + 1, 0.01),
                     np.arange(X_fixed[:, 1].min() - 1, X_fixed[:, 1].max() + 1, 0.01))

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) - b
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])

# Graficar el hiperplano
plt.plot(xx, (-w[0] * xx - b) / w[1], 'k--')
plt.title('Máquinas de Soporte Vectorial')
plt.legend()
plt.grid()
plt.show()

X_test = [-2, -4]
prediccion = prediccion_MSV(X_test, w, b)
print(f"La clase del valor es {prediccion}")
