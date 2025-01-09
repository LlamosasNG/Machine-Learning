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

# Cargar los datos del archivo CSV
ruta_archivo = './train.csv'
datos = pd.read_csv(ruta_archivo)

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

