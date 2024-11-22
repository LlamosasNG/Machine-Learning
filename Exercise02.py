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
  plt.title(title)
  plt.show()

def svd(matriz):
  U, S, Vt = np.linalg.svd(matriz)
  return U, S, Vt

A = np.array([[1,2],[2,1]])
U, S, Vt = svd(A)
print("U:\n", U)
print("S:\n", S)
print("Vt:\n", Vt)

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

visualize_patterns([base_A], "Patrón Base A")
visualize_patterns([base_1], "Patrón Base 1")
visualize_patterns(patterns_A, "Variaciones de Patrón A")
visualize_patterns(patterns_1, "Variaciones de Patrón 1")
visualize_patterns([noisy_patterns_A], "Patrón Ruido A")
visualize_patterns([noisy_patterns_1], "Patrón Ruido 1")