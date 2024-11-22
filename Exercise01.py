import numpy as np
import matplotlib.pyplot as plt

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

def distancia_ecluidiana(X, x_test):
    d = (X - x_test)**2
    distancia = [None] * len(d)
    for i in range(len(d)):
        distancia[i] = (np.sqrt(d[i][0] + d[i][1]), i) 
    return distancia

def KNN(K, lista_con_indice_ordenados, Y_etiquetas):
    contador_clase1 = 0
    contador_clase2 = 0
    for j in range(0, K):
        index = lista_con_indice_ordenados[j][1]
        if Y[index] == 0:
            contador_clase1 += 1
        elif Y[index] == 1:
            contador_clase2 += 1
    if contador_clase1 > contador_clase2:
        print("Eres Hombre")
        return 0
    else:
        print("Eres Mujer")
        return 1

def graficar_datos(X, Y, x_test, Clase_X_test, vecinos_cercanos):
    for i in range(len(X)):
        if Y[i] == 0:
            plt.scatter(X[i][0], X[i][1], marker='^', color='red', label='Clase 0' if i == 0 else '')
        else:
            plt.scatter(X[i][0], X[i][1], marker='o', color='blue', label='Clase 1' if i == 1 else '')
    if Clase_X_test == 0:
        plt.scatter(x_test[0], x_test[1], marker='^', color='red', s=100, label='Clase 0 X_test')
    else:
        plt.scatter(x_test[0], x_test[1], marker='o', color='blue', s=100, label='Clase 1 X_test')

    for vecino in vecinos_cercanos:
        vecino_pos = X[vecino[1]]
        plt.plot([x_test[0], vecino_pos[0]], [x_test[1], vecino_pos[1]], 'k--', linewidth=0.5)

    plt.xlabel('Altura')
    plt.ylabel('Temperatura')
    plt.legend()
    plt.title('Clasificador KNN')
    plt.grid()
    plt.show()

X = np.array([
    [1.75, 37.1],
    [1.81, 37.2],
    [1.72, 36.9],
    [1.77, 37.2],
    [1.76, 36.8],
    [1.55, 36.6],
    [1.59, 36.7],
    [1.61, 36.5],
    [1.62, 36.4],
    [1.60, 36.3]
])
Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
K = 3

# Paso 1: Dado un nuevo punto x_test, calcular la distancia entre x_test y todos los puntos de entrenamiento
X_test0 = float(input("Dame el valor de Altura: "))
X_test1 = float(input("Dame el valor de Temperatura: "))
X_test = np.array([X_test0, X_test1])

distancias = distancia_ecluidiana(X, X_test)

# Paso 2: Ordenar las distancias de menor a mayor usando quick_sort
quick_sort(distancias, 0, len(distancias) - 1)

# Paso 3: Calculas los K-ésimos más cercanos
Clase_X_Test = KNN(K, distancias, Y)

# Paso 4 (opcional pero altamente recomendado), graficar
graficar_datos(X, Y, X_test, Clase_X_Test, distancias[:K])
