import matplotlib.pyplot as plt
import random
import numpy as np

def distancia_ecluidiana(X, x_test):
    x_test = np.array(x_test)
    d = (X - x_test) ** 2
    distancia = [np.sqrt(d[i][0] + d[i][1]) for i in range(len(d))]
    return distancia

def inicializar_centroides(data, k):
    data = np.array(data)
    num_datos = len(data)
    
    # Inicializar el array de centroides con tamaño k x num_caracteristicas
    num_caracteristicas = data.shape[1]
    centroides = np.zeros((k, num_caracteristicas))
    
    # Seleccionar el primer centroide al azar
    primer_centroide_idx = random.randint(0, num_datos - 1)
    centroides[0] = data[primer_centroide_idx]
    
    for i in range(1, k):
        # Calcular las distancias mínimas al conjunto actual de centroides
        distancias_minimas = np.zeros(len(data))  # Crear un array inicializado con ceros
        for idx, punto in enumerate(data):
            distancias = [np.linalg.norm(punto - centroides[j]) for j in range(i)]
            distancias_minimas[idx] = min(distancias)  # Asignar el mínimo directamente al índice correspondiente

        # Elegir el siguiente centroide con probabilidad proporcional al cuadrado de la distancia
        distancias_cuadradas = distancias_minimas ** 2
        probabilidades = distancias_cuadradas / distancias_cuadradas.sum()
        siguiente_centroide_idx = np.random.choice(range(len(data)), p=probabilidades)
        centroides[i] = data[siguiente_centroide_idx]

    return centroides

def Kmeans(data, k, epocas):
    data = np.array(data)
    num_datos, num_caracteristicas = data.shape

    # Inicializar centroides usando K-Means++
    centroides = inicializar_centroides(data, k)

    # Iteraciones para calcular los clústeres y actualizar los centroides
    for iteraciones in range(epocas):
        clouster_asignados = [0] * num_datos
        for i in range(num_datos):
            distancias = [distancia_ecluidiana([centroide], data[i])[0] for centroide in centroides]
            clouster_asignados[i] = np.argmin(distancias)

        # Actualizar centroides
        Nuevos_Centroides = [[0] * num_caracteristicas for _ in range(k)]
        puntos_por_clouster = [0] * k
        for i in range(num_datos):
            clouster = clouster_asignados[i]
            puntos_por_clouster[clouster] += 1
            for j in range(num_caracteristicas):
                Nuevos_Centroides[clouster][j] += data[i][j]

        for clouster_index in range(k):
            if puntos_por_clouster[clouster_index] > 0:
                for j in range(num_caracteristicas):
                    Nuevos_Centroides[clouster_index][j] /= puntos_por_clouster[clouster_index]
            else:
                for j in range(num_caracteristicas):
                    Nuevos_Centroides[clouster_index][j] = centroides[clouster_index][j]

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
        clousters[clouster_asignados[i]] = clousters[clouster_asignados[i]] + [data[i]] 
    
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
    plt.title("K-means++")
    plt.legend()
    plt.grid(True)
    plt.show()


# Datos de prueba dispersos
data = [
    # Clase 1: Alrededor de (-50, -50)
    [random.uniform(-55, -35), random.uniform(-55, -35)] for _ in range(20)
] + [
    # Clase 2: Alrededor de (-20, 20)
    [random.uniform(-25, -5), random.uniform(15, 35)] for _ in range(20)
] + [
    # Clase 3: Alrededor de (0, -30)
    [random.uniform(-5, 15), random.uniform(-35, -15)] for _ in range(20)
] + [
    # Clase 4: Alrededor de (40, 40)
    [random.uniform(35, 55), random.uniform(35, 55)] for _ in range(20)
] + [
    # Clase 5: Alrededor de (60, -60)
    [random.uniform(55, 75), random.uniform(-65, -45)] for _ in range(20)
] + [
    # Clase 6: Alrededor de (-70, 70)
    [random.uniform(-75, -55), random.uniform(65, 85)] for _ in range(20)
] + [
    # Clase 7: Alrededor de (80, 10)
    [random.uniform(75, 95), random.uniform(5, 25)] for _ in range(20)
] + [
    # Clase 8: Alrededor de (-90, -90)
    [random.uniform(-95, -75), random.uniform(-95, -75)] for _ in range(20)
]

k = 8
epocas = 100
centroides, clousters = Kmeans(data, k, epocas)
graficar_puntos(data, clousters, centroides)