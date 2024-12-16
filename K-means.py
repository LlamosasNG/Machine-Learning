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
    centroides = [[0] * num_caracteristicas for _ in range(k)]
    for i in range(k):
        while True:
            random_centroides = random.randint(0, numero_datos - 1)
            if random_centroides not in indices_usados:
                indices_usados[i] = random_centroides
                for j in range(num_caracteristicas):
                    centroides[i][j] = data[random_centroides][j]
                break

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
    clousters = [[[0] * num_caracteristicas for _ in range(num_datos)] for _ in range(k)]
    puntos_por_clouster = [0] * k
    for i in range(num_datos):
        clouster = clouster_asignados[i]
        clousters[clouster][puntos_por_clouster[clouster]] = data[i]
        puntos_por_clouster[clouster] += 1

    # Reducir el tamaño de los clústeres eliminando puntos vacíos
    clousters_final = []
    for clouster_index in range(k):
        clousters_final.append(clousters[clouster_index][:puntos_por_clouster[clouster_index]])

    return centroides, clousters_final

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

data = [
    # Clase 1: Alrededor de (-50, -50)
    [-50, -50], [-48, -52], [-55, -45], [-53, -48], [-47, -55],
    [-54, -51], [-49, -49], [-52, -54], [-48, -47], [-50, -53],
    [-51, -49], [-53, -52], [-47, -51], [-50, -47], [-48, -50],
    [-52, -50], [-51, -54], [-49, -53], [-54, -49], [-47, -48],

    # Clase 2: Alrededor de (-20, 20)
    [-20, 20], [-18, 22], [-25, 25], [-23, 18], [-17, 15],
    [-24, 19], [-19, 21], [-22, 24], [-18, 17], [-20, 23],
    [-21, 19], [-23, 22], [-17, 21], [-20, 17], [-18, 20],
    [-22, 20], [-21, 24], [-19, 23], [-24, 19], [-17, 18],

    # Clase 3: Alrededor de (0, -30)
    [0, -30], [2, -32], [-5, -35], [-3, -28], [3, -25],
    [-4, -29], [-1, -31], [-2, -34], [2, -27], [0, -33],
    [-1, -29], [-3, -32], [3, -31], [0, -27], [2, -30],
    [-2, -30], [-1, -34], [1, -33], [-4, -29], [3, -28],

    # Clase 4: Alrededor de (40, 40)
    [40, 40], [42, 38], [35, 45], [43, 48], [37, 35],
    [44, 41], [39, 39], [42, 44], [38, 37], [40, 43],
    [41, 39], [43, 42], [37, 41], [40, 37], [38, 40],
    [42, 40], [41, 44], [39, 43], [44, 39], [37, 38],

    # Clase 5: Alrededor de (60, -60)
    [60, -60], [58, -62], [65, -65], [63, -58], [57, -55],
    [64, -59], [59, -61], [62, -64], [58, -57], [60, -63],
    [61, -59], [63, -62], [57, -61], [60, -57], [58, -60],
    [62, -60], [61, -64], [59, -63], [64, -59], [57, -58],

    # Clase 6: Alrededor de (-70, 70)
    [-70, 70], [-68, 72], [-75, 75], [-73, 68], [-67, 65],
    [-74, 69], [-69, 71], [-72, 74], [-68, 67], [-70, 73],
    [-71, 69], [-73, 72], [-67, 71], [-70, 67], [-68, 70],
    [-72, 70], [-71, 74], [-69, 73], [-74, 69], [-67, 68],

    # Clase 7: Alrededor de (80, 10)
    [80, 10], [78, 12], [85, 15], [83, 8], [77, 5],
    [84, 9], [79, 11], [82, 14], [78, 7], [80, 13],
    [81, 9], [83, 12], [77, 11], [80, 7], [78, 10],
    [82, 10], [81, 14], [79, 13], [84, 9], [77, 8],

    # Clase 8: Alrededor de (-90, -90)
    [-90, -90], [-88, -92], [-95, -95], [-93, -88], [-87, -85],
    [-94, -89], [-89, -91], [-92, -94], [-88, -87], [-90, -93],
    [-91, -89], [-93, -92], [-87, -91], [-90, -87], [-88, -90],
    [-92, -90], [-91, -94], [-89, -93], [-94, -89], [-87, -88],
]

k = 8
epocas = 100
centroides, clousters = Kmeans(data, k, epocas)
graficar_puntos(data, clousters, centroides) 
