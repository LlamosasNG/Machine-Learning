import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
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
    num_caracteristicas = data.shape[1]
    
    # Inicializar la matriz de centroides
    centroides = np.zeros((k, num_caracteristicas))
    
    # Seleccionar el primer centroide aleatoriamente
    primer_centroide_idx = random.randint(0, num_datos - 1)
    centroides[0] = data[primer_centroide_idx]
    
    for i in range(1, k):
        # Calcular las distancias mínimas al conjunto actual de centroides
        distancias_minimas = []
        for punto in data:
            distancias = [np.linalg.norm(punto - centroides[j]) for j in range(i)]
            distancias_minimas.append(min(distancias))
        
        # Elegir el siguiente centroide con probabilidad proporcional a las distancias al cuadrado
        distancias_cuadradas = np.array(distancias_minimas) ** 2
        probabilidades = distancias_cuadradas / np.sum(distancias_cuadradas)
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

def graficar_clustering_3D(data, clousters, centroides):
    colors = ['red', 'blue']
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar puntos de cada clúster
    for clouster_index, clouster in enumerate(clousters):
        clouster = np.array(clouster)
        if len(clouster) > 0:  # Verificar que el clúster no esté vacío
            ax.scatter(
                clouster[:, 0],  # Pclass
                clouster[:, 1],  # Age
                clouster[:, 2],  # Sex
                color=colors[clouster_index % len(colors)],
                label=f'Clúster {clouster_index + 1}'
            )

    # Graficar los centroides
    centroides = np.array(centroides)
    ax.scatter(
        centroides[:, 0],  # Pclass
        centroides[:, 1],  # Age
        centroides[:, 2],  # Sex
        color='black',
        marker='x',
        s=100,
        label='Centroides'
    )

    # Etiquetas y leyenda
    ax.set_xlabel('Pclass')
    ax.set_ylabel('Age')
    ax.set_zlabel('Sex')
    ax.set_title('Clasificación 3D con K-Means++')
    ax.legend()
    plt.show()

# Cargar los datos del archivo CSV
ruta_archivo = './train.csv'  # Reemplazar con la ruta de tu archivo
datos = pd.read_csv(ruta_archivo).head(100)

# Extraer las columnas relevantes
Y = datos['Survived'].tolist()
Sex = datos['Sex'].tolist()
Age = datos['Age'].tolist()
Pclass = datos['Pclass'].tolist()

# Procesar valores faltantes o inconsistentes
moda_sex = max(set(Sex), key=Sex.count)  
Sex = [moda_sex if pd.isna(s) else s for s in Sex]
Sex = [1 if s == 'male' else 0 for s in Sex]  # 1: Hombre, 0: Mujer

Age = [np.nan if pd.isna(a) else a for a in Age]
promedio_edad = np.nanmean(Age)
Age = [promedio_edad if np.isnan(a) else a for a in Age]

moda_clase = max(set(Pclass), key=Pclass.count)
Pclass = [moda_clase if pd.isna(c) else c for c in Pclass]
Pclass = [int(c) for c in Pclass]

# Crear la matriz de características X
X = np.array(list(zip(Pclass, Age, Sex)))

# Ejecutar el algoritmo K-Means++
k = 2 
epocas = 100
centroides, clousters = Kmeans(X, k, epocas)

# Llamar a la función para graficar
graficar_clustering_3D(X, clousters, centroides)

# Imprimir resultados
print("Centroides:")
for i, centroide in enumerate(centroides):
    print(f"Clúster {i + 1}: {centroide}")

print("\nPuntos en cada clúster:")
for i, clouster in enumerate(clousters):
    print(f"Clúster {i + 1}: {len(clouster)} puntos")