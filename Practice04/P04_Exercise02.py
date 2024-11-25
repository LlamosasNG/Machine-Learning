import numpy as np
import matplotlib.pyplot as plt

# Definición de la función para visualizar patrones
def  visualize_patterns(patterns, title):
  num_patterns = len(patterns)
  fig, axs = plt.subplots(1, num_patterns, figsize=(12, 6))
  if num_patterns == 1:
    axs = [axs]
  for i in range(num_patterns):
    axs[i].imshow(patterns[i], cmap='gray')
    axs[i].axis('off')
  plt.suptitle(title)
  plt.show()

def svd(matriz):
  U, S, Vt = np.linalg.svd(matriz)
  return U, S, Vt

# Función para determinar distancias euclidianas
def distancia_ecluidiana(X, x_test):
    distancias = [0] * len(X)
    for i in range(len(X)):
        distancias[i] = np.sqrt(np.sum((X[i] - x_test) ** 2))
    return distancias

# Algoritmo KNN
def KNN(K, lista_con_indice_ordenados, Y):
  clase1 = 0
  clase2 = 0
  
  for j in range(K):
    index = lista_con_indice_ordenados[j][1]
    if Y[index] == 0:
      clase1 += 1
    elif Y[index] == 1:
      clase2 += 1
  
  return 0 if clase1 > clase2 else 1

# Ordenar lista con posiciones
def ordenar_lista_con_posiciones(lista):
  lista_con_indices = [(lista[i], i) for i in range(len(lista))]
  lista_con_indices.sort(key=lambda x: x[0])  
  
  return lista_con_indices
  
# Definición manual de los patrones base
base_A = np.zeros((10,10))
base_A[1:9, 3] = 1
base_A[1:9, 6] = 1
base_A[4, 3:7] = 1

base_1 = np.zeros((10,10))
base_1[1:9, 4:6] = 1

# Generación manual de variaciones de los patrones
patterns_A = [base_A + np.random.uniform(-0.2, 0.2, base_A.shape) for _ in range(5)]
patterns_1 = [base_1 + np.random.uniform(-0.2, 0.2, base_1.shape) for _ in range(5)]

# Generación de patrones ruidosos
noisy_patterns_A = base_A + np.random.uniform(-0.5, 0.5, base_A.shape)
noisy_patterns_1 = base_1 + np.random.uniform(-0.5, 0.5, base_1.shape)

# Aplicar SVD a todos los patrones
dimensiones = 5
X = []

for pattern in patterns_A + patterns_1:
  U, S, Vt = svd(pattern)
  X += [U[:, :dimensiones] @ np.diag(S[:dimensiones])]
  
X = np.array(X)
Y = [0] * 5 + [1] * 5

# Representaciones de los patrones ruidosos
U_A, S_A, _ = svd(noisy_patterns_A)
test_A = U_A[:, :dimensiones] @ np.diag(S_A[:dimensiones]) 

U_1, S_1, _ = svd(noisy_patterns_1)
test_1 = U_1[:, :dimensiones] @ np.diag(S_1[:dimensiones]) 

# Clasificación usando KNN
distancias_A = distancia_ecluidiana(X, test_A)
distancias_1 = distancia_ecluidiana(X, test_1)

ordenados_A = ordenar_lista_con_posiciones(distancias_A)
ordenados_1 = ordenar_lista_con_posiciones(distancias_1)

K = 3
clase_A = KNN(K, ordenados_A, Y)
clase_1 = KNN(K, ordenados_1, Y)

visualize_patterns([base_A], "Patrón Base A")
visualize_patterns([base_1], "Patrón Base 1")
visualize_patterns(patterns_A, "Variaciones de Patrón A")
visualize_patterns(patterns_1, "Variaciones de Patrón 1")
visualize_patterns([noisy_patterns_A], f"Patrón Ruido A - Clasificado como {'A' if clase_A == 0 else '1'}")
visualize_patterns([noisy_patterns_1], f"Patrón Ruido 1 - Clasificado como {'1' if clase_1 == 0 else '1'}")