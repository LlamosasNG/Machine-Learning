import numpy as np
import matplotlib.pyplot as plt

# Diccionario de mapeo de clases a letras
class_map = {0: "O", 1: "T", 2: "L", 3: "J"}

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

# SVD
def svd(matriz):
    U, S, Vt = np.linalg.svd(matriz)
    return U, S, Vt

# Distancia Euclidiana
def distancia_euclidiana(X, x_test):
    distancias = [(np.sqrt(np.sum((X[i] - x_test) ** 2)), i) for i in range(len(X))]
    return distancias

# KNN
def KNN(K, lista_con_indice_ordenados, Y):
    conteo = [0, 0, 0, 0]
    for j in range(K):
        index = lista_con_indice_ordenados[j][1]
        conteo[Y[index]] += 1
    return np.argmax(conteo)

# Implementación de QuickSort
def quick_sort(A, p, r):
    if p < r:
        j = pivot(A, p, r)
        quick_sort(A, p, j - 1)
        quick_sort(A, j + 1, r)

def pivot(A, p, r):
    piv = A[p][0]
    i = p + 1
    j = r

    while True:
        while i <= r and A[i][0] <= piv:
            i += 1
        while A[j][0] > piv:
            j -= 1
        if i >= j:
            break
        A[i], A[j] = A[j], A[i]
    A[p], A[j] = A[j], A[p]
    return j
  
# Patrones Base
base_O = np.zeros((10, 10))
base_O[1, 3:7] = 1
base_O[8, 3:7] = 1
base_O[2:8, 3] = 1
base_O[2:8, 6] = 1

base_T = np.zeros((10, 10))
base_T[1, 2:8] = 1
base_T[2:9, 5] = 1

base_L = np.zeros((10, 10))
base_L[1:9, 3] = 1
base_L[8, 3:7] = 1

base_J = np.zeros((10, 10))
base_J[1:9, 5] = 1
base_J[8, 3:6] = 1
base_J[7:9, 3] = 1

# Generación de Variaciones
patterns_O = [base_O + np.random.uniform(-0.1, 0.1, base_O.shape) for _ in range(10)]
patterns_T = [base_T + np.random.uniform(-0.1, 0.1, base_T.shape) for _ in range(10)]
patterns_L = [base_L + np.random.uniform(-0.1, 0.1, base_L.shape) for _ in range(10)]
patterns_J = [base_J + np.random.uniform(-0.1, 0.1, base_J.shape) for _ in range(10)]

# Generación de Patrones Ruidosos
noisy_patterns_O = base_O + np.random.uniform(-0.2, 0.2, base_O.shape)
noisy_patterns_T = base_T + np.random.uniform(-0.2, 0.2, base_T.shape)
noisy_patterns_L = base_L + np.random.uniform(-0.2, 0.2, base_L.shape)
noisy_patterns_J = base_J + np.random.uniform(-0.2, 0.2, base_J.shape)

# Aplicar SVD y Reducir Dimensionalidad
dimensiones = 5
X = []

for pattern in patterns_O + patterns_T + patterns_L + patterns_J:
    U, S, Vt = svd(pattern)
    X.append(U[:, :dimensiones] @ np.diag(S[:dimensiones]))

X = np.array(X)

# Etiquetas para todas las clases
Y = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 

# Representaciones de Patrones Ruidosos
U_A, S_A, _ = svd(noisy_patterns_O)
test_O = U_A[:, :dimensiones] @ np.diag(S_A[:dimensiones])

U_T, S_T, _ = svd(noisy_patterns_T)
test_T = U_T[:, :dimensiones] @ np.diag(S_T[:dimensiones])

U_L, S_L, _ = svd(noisy_patterns_L)
test_L = U_L[:, :dimensiones] @ np.diag(S_L[:dimensiones])

U_J, S_J, _ = svd(noisy_patterns_J)
test_J = U_J[:, :dimensiones] @ np.diag(S_J[:dimensiones])

# Clasificación usando KNN
distancias_O = distancia_euclidiana(X, test_O)
distancias_T = distancia_euclidiana(X, test_T)
distancias_L = distancia_euclidiana(X, test_L)
distancias_J = distancia_euclidiana(X, test_J)

quick_sort(distancias_O, 0, len(distancias_O)-1)
quick_sort(distancias_T, 0, len(distancias_T)-1)
quick_sort(distancias_L, 0, len(distancias_L)-1)
quick_sort(distancias_J, 0, len(distancias_J)-1)

K = 5
clase_O = KNN(K, distancias_O, Y)
clase_T = KNN(K, distancias_T, Y)
clase_L = KNN(K, distancias_L, Y)
clase_J = KNN(K, distancias_J, Y)

# Visualizar Patrones
visualize_patterns([base_O], "Patrón Base O")
visualize_patterns([base_T], "Patrón Base T")
visualize_patterns([base_L], "Patrón Base L")
visualize_patterns([base_J], "Patrón Base J")
visualize_patterns(patterns_O, "Variaciones de Patrón O")
visualize_patterns(patterns_T, "Variaciones de Patrón T")
visualize_patterns(patterns_L, "Variaciones de Patrón L")
visualize_patterns(patterns_J, "Variaciones de Patrón J")
visualize_patterns([noisy_patterns_O], f"Patrón Ruido O - Clasificado como {class_map.get(clase_O, 'Desconocido')}")
visualize_patterns([noisy_patterns_T], f"Patrón Ruido T - Clasificado como {class_map.get(clase_T, 'Desconocido')}")
visualize_patterns([noisy_patterns_L], f"Patrón Ruido L - Clasificado como {class_map.get(clase_L, 'Desconocido')}")
visualize_patterns([noisy_patterns_J], f"Patrón Ruido J - Clasificado como {class_map.get(clase_J, 'Desconocido')}")