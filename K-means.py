import matplotlib.pyplot as plt
import random
import numpy as np

def distancia_ecluidiana(X, x_test):
    x_test = np.array(x_test)
    d = (X - x_test) ** 2
    distancia = [np.sqrt(d[i][0] + d[i][1]) for i in range(len(d))]
    return distancia

def Kmeans(data, k, epocas):
    data = np.array(data)
    numero_datos = len(data)
    num_datos, num_caracteristicas = data.shape

    # Inicializar aleatoriamente los centroides sin duplicados
    indices_usados = [-1] * k
    centroides = []
    for i in range(k):
        while True:
            random_centroides = random.randint(0, numero_datos - 1)
            if random_centroides not in indices_usados:
                indices_usados[i] = random_centroides
                centroides.append(data[random_centroides])
                break

    # Iteraciones para calcular los clústeres y actualizar los centroides
    for iteraciones in range(epocas):
        clouster_asignados = [0] * num_datos
        for i in range(num_datos):
            distancias = [distancia_ecluidiana([centroide], data[i])[0] for centroide in centroides]
            clouster_asignados[i] = np.argmin(distancias)

        # Actualizar centroides
        Nuevos_Centroides = []
        for clouster_index in range(k):
            puntos_en_clouster = [data[i] for i in range(num_datos) if clouster_asignados[i] == clouster_index]

            if len(puntos_en_clouster) > 0:
                clouster_sumas = [0] * num_caracteristicas
                for punto in puntos_en_clouster:
                    for j in range(num_caracteristicas):
                        clouster_sumas[j] += punto[j]
                promedio_clouster = [clouster_sumas[j] / len(puntos_en_clouster) for j in range(num_caracteristicas)]
                Nuevos_Centroides.append(promedio_clouster)
            else:
                Nuevos_Centroides.append(centroides[clouster_index])

        # Verificar convergencia
        variacion = True
        epsilon = 1e-10
        for i in range(k):
            for j in range(num_caracteristicas):
                if abs(Nuevos_Centroides[i][j] - centroides[i][j]) > epsilon:
                    variacion = False
                    break
            if not variacion:
                break

        if variacion:
            break

        centroides = Nuevos_Centroides

    # Agrupamiento final
    clousters = [[] for _ in range(k)]
    for i in range(num_datos):
        clousters[clouster_asignados[i]].append(data[i])

    return centroides, clousters

def graficar_puntos(data, clousters, centroides):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'magenta']
    for clouster_index in range(len(clousters)):
        x_points = [clousters[clouster_index][i][0] for i in range(len(clousters[clouster_index]))]
        y_points = [clousters[clouster_index][i][1] for i in range(len(clousters[clouster_index]))]
        plt.scatter(x_points, y_points, color=colors[clouster_index % len(colors)], label=f'Clúster {clouster_index + 1}')
    x_centroides = [centroides[i][0] for i in range(len(centroides))]
    y_centroides = [centroides[i][1] for i in range(len(centroides))]
    plt.scatter(x_centroides, y_centroides, color='black', marker='x', s=100, label='Centroides')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Kmeans")
    plt.legend()
    plt.grid(True)
    plt.show()

# Datos de prueba
data = [
    [1, 2], [1.5, 2.3], [1.2, 1.9],  # Clase 1
    [4, 5], [4.1, 5.1], [4.4, 5.3],  # Clase 2
    [9, 10], [9.1, 10.1], [8.8, 10.4],  # Clase 3
    [15, 16.1], [15.2, 16.5], [14.9, 15.9],  # Clase 4
    [20, 5], [20.3, 5.1], [19.8, 5.2],  # Clase 5
    [5, 20], [5.2, 20.3], [4.8, 19.7],  # Clase 6
    [25, 25], [24.9, 25.1], [25.1, 24.8],  # Clase 7
    [10, 30], [10.2, 30.1], [9.8, 30.3]  # Clase 8
]

k = 8
epocas = 100
centroides, clousters = Kmeans(data, k, epocas)
graficar_puntos(data, clousters, centroides)
