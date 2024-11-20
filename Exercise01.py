import numpy as np
import matplotlib.pyplot as plt

def ordenar_lista_con_posiciones(lista):
    lista_con_indices = []
    for i in range(len(lista)):
        lista_con_indices += [(lista[i], i)]

    n = len(lista_con_indices)

    for i in range(n):
        minimo=lista_con_indices[i]
        min_index = i

        for j in range(i+1,n):
            if lista_con_indices[j][0] < minimo[0]:
                minimo=lista_con_indices[j]
                min_index = j
        lista_con_indices[i], lista_con_indices[min_index]= lista_con_indices[min_index], lista_con_indices[i]\
            
    return lista_con_indices

def distancia_ecluidiana(X,x_test):
    ## X puede un punto o un vector de dos posiciones
    ## x_test es un solo punto a clasificar
    d = (X-x_test)**2
    distancia = [None] * len(d)
    for i in range(len(d)):
        distancia[i]= np.sqrt(d[i][0]+d[i][1])
    return distancia

def KNN(K, lista_con_indice_ordenados):
    contador_clase1 = 0
    contador_clase2 = 0
    for j in range(0,K):
        index = lista_con_indice_ordenados[j][1]
        if Y[index]==0:
            contador_clase1 += 1
        elif Y[index]==1:
            contador_clase2 += 1
    if contador_clase1 > contador_clase2:
        print("Eres Hombre")
        return 0
    else:
        print("Eres Mujer")
        return 1

def graficar_datos(X, Y, x_test, Clase_X_test, vecinos_cercanos):
    for i in range(len(X)):
        if Y[i]==0:
            plt.scatter(X[i][0],X[i][1], marker='^', color='red', label='Clase 0' if i == 0 else '')  ## if i == 0 else '': Le da la misma etiqueta a todos
        else:
            plt.scatter(X[i][0], X[i][1], marker='o', color='blue', label='Clase 1' if i == 1 else '')  ## if i == 1 else '': Le da la misma etiqueta a todos
    if Clase_X_test == 0:
        plt.scatter(x_test[0], x_test[1], marker='^', color='red', s= 100 ,label='Clase 0 X_test')
    else:
        plt.scatter(x_test[0], x_test[1], marker='o', color='blue', s= 100, label='Clase 1 X_test')

    for vecino in vecinos_cercanos:
        vecino_pos = X[vecino[1]]
        plt.plot([x_test[0],vecino_pos[0]], [x_test[1],vecino_pos[1]], 'k--', linewidth=0.5)

    plt.xlabel('Altura')
    plt.ylabel('Temperatura')
    plt.legend()
    plt.title('Clasificador KNN')
    plt.grid()
    plt.show()

X = [[1.75,37.1],
    [1.81,37.2],
    [1.72,36.9],
    [1.77,37.2],
    [1.76,36.8],
    [1.55,36.6],
    [1.59,36.7],
    [1.61,36.5],
    [1.62,36.4],
    [1.60,36.3]]
Y = [0,0,0,0,0,1,1,1,1,1]
K = 3

## Paso 1 Dado un nuevo punto x_test, calcular la distancia entre x_test y todos los puntos de entrenamiento
X_test0 = float(input("Dame el valor de Altura:"))
X_test1 = float(input("Dame el valor de Temperatura:"))
X_test = np.array([X_test0, X_test1])

distancias = distancia_ecluidiana(X,X_test)

## Paso 2 Ordenar las distancias de menor a mayor
dist_ordenada_indi = ordenar_lista_con_posiciones(distancias)

## Paso 3 Calculas los K-esimos mas cercanos
Clase_X_Test = KNN(K,dist_ordenada_indi,Y)

## paso 4 (opcional pero altamente recomendado), graficar
graficar_datos(X, Y, X_test, Clase_X_Test, dist_ordenada_indi[:K])

