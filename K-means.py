import matplotlib.pyplot as plt
import numpy as np
import random

def distancia_euclidiana(X, centroids):
    # Calcula distancias entre cada punto en X y cada centroide, vectorizado
    return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

def Kmeans(data, k, epocas):
    data = np.array(data)
    num_datos, num_caracteristicas = data.shape

    # Inicializar centroides aleatoriamente sin duplicados
    random_indices = random.sample(range(num_datos), k)
    centroides = data[random_indices]

    for iteraciones in range(epocas):
        # Asignar cada punto al clúster más cercano
        distancias = distancia_euclidiana(data, centroides)
        clouster_asignados = np.argmin(distancias, axis=1)

        # Actualizar centroides calculando el promedio de cada clúster
        nuevos_centroides = np.zeros_like(centroides)
        for i in range(k):
            puntos_en_clouster = data[clouster_asignados == i]
            if len(puntos_en_clouster) > 0:
                nuevos_centroides[i] = puntos_en_clouster.mean(axis=0)
            else:
                nuevos_centroides[i] = centroides[i]  # Si el clúster está vacío, conservar el centroide actual

        # Verificar convergencia (Early Stopping)
        if np.allclose(centroides, nuevos_centroides, atol=1e-6):
            break

        centroides = nuevos_centroides

    # Agrupar datos en clústeres finales
    clousters = [[] for _ in range(k)]
    for i in range(num_datos):
        clousters[clouster_asignados[i]].append(data[i])

    return centroides, clousters

def graficar_puntos(data, centroides=None):
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], color='red', label='Datos')
    if centroides is not None:
        centroides = np.array(centroides)
        plt.scatter(centroides[:, 0], centroides[:, 1], color='blue', label='Centroides', marker='x', s=100)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Kmeans")
    plt.grid()
    plt.legend()
    plt.show()

# Datos de prueba
data = [
    [1, 2], [1.5, 2.3], [1.2, 1.9],
    [4, 5], [4.1, 5.1], [4.4, 5.3], 
    [9, 10], [9.1, 10.1], [8.8, 10.4],
    [15, 16.1], [15.2, 16.5], [14.9, 15.9]
]

k = 4
epocas = 100
centroides, clousters = Kmeans(data, k, epocas)

# Imprimir resultados
print("Centroides finales:", centroides)
print("Clústeres:")
for i, clouster in enumerate(clousters):
    print(f"Clúster {i+1}: {clouster}")

# Graficar resultados
graficar_puntos(data, centroides)
