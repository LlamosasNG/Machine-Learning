import numpy as np
import matplotlib.pyplot as plt

# Entrenamiento SVM
def entrenamiento_MSV(X, Y):
    numero_muestras, numero_caracteristicas = X.shape

    # Inicialización parámetros externos
    epocas = 1000
    lr = 0.0001
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
    tolerancia = .0001
    vectores_soporte_indices = [
        i for i, x in enumerate(X)
        if abs(Y[i] * (np.dot(x, w) + b) - 1) <= tolerancia
    ]
    vectores_soporte = X[vectores_soporte_indices]

    return w, b, vectores_soporte, vectores_soporte_indices

# Predicción SVM
def prediccion_MSV(X_test, w, b):
    return np.sign(np.dot(X_test, w) + b)

# Función para visualizar patrones
def visualize_patterns(patterns, title):
    num_patterns = len(patterns)
    fig, axs = plt.subplots(1, num_patterns, figsize=(12, 6))
    if num_patterns == 1:
        axs = [axs]
    for i in range(num_patterns):
        axs[i].imshow(patterns[i], cmap='gray')
        axs[i].axis('off')
    plt.suptitle(title)
    plt.show()

# Definición manual de los patrones base
base_T = np.zeros((10, 10))
base_T[1, 2:8] = 1
base_T[2:9, 5] = 1

base_J = np.zeros((10, 10))
base_J[1:9, 5] = 1
base_J[8, 3:6] = 1
base_J[7:9, 3] = 1

# Generación manual de variaciones de los patrones
patterns_T = [base_T + np.random.uniform(-0.05, 0.05, base_T.shape) for _ in range(20)]
patterns_J = [base_J + np.random.uniform(-0.05, 0.05, base_J.shape) for _ in range(20)]

noisy_patterns_T = base_T + np.random.uniform(-0.05, 0.05, base_T.shape)
noisy_patterns_J = base_J + np.random.uniform(-0.05, 0.05, base_J.shape)

# Aplicar SVD a todos los patrones
dimensiones = 5
X = []

for pattern in patterns_T + patterns_J:
    U, S, Vt = np.linalg.svd(pattern)
    X.append((U[:, :dimensiones] @ np.diag(S[:dimensiones])).flatten())  # Aplanar

X = np.array(X)
Y = np.array([-1] * 20 + [1] * 20)  # Clases: -1 para A, 1 para 1

# Representaciones de los patrones ruidosos
U_A, S_A, _ = np.linalg.svd(noisy_patterns_T)
test_T = (U_A[:, :dimensiones] @ np.diag(S_A[:dimensiones])).flatten()

U_1, S_1, _ = np.linalg.svd(noisy_patterns_J)
test_J = (U_1[:, :dimensiones] @ np.diag(S_1[:dimensiones])).flatten()

# Entrenamiento con SVM
w, b, vectores_soporte, vectores_soporte_indices = entrenamiento_MSV(X, Y)

# Predicción
clase_T = prediccion_MSV(test_T, w, b)
clase_J = prediccion_MSV(test_J, w, b)

print(clase_T)
print(clase_J)

# Visualización
visualize_patterns([base_T], "Patrón Base T")
visualize_patterns([base_J], "Patrón Base J")
visualize_patterns(patterns_T, "Variaciones de Patrón T")
visualize_patterns(patterns_J, "Variaciones de Patrón J")
visualize_patterns([noisy_patterns_T], f"Patrón Ruido T - Clasificado como {'T' if clase_T == -1 else 'J'}")
visualize_patterns([noisy_patterns_J], f"Patrón Ruido J - Clasificado como {'T' if clase_J == -1 else 'J'}")

