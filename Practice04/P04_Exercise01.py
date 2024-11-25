import pandas as pd
import numpy as np

# Función para determinar distancias euclidianas
def distancia_euclidiana(X, x_test):
    distancias = [(np.sqrt(np.sum((X[i] - x_test) ** 2)), i) for i in range(len(X))]
    return distancias

# Implementación de QuickSort
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

# Implementación de KNN
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

# Entrada del usuario
edad_usuario = float(input("Ingrese la edad: "))
sexo_usuario = input("Ingrese el sexo (male/female): ")
clase_usuario = int(input("Ingrese la clase (1, 2 o 3): "))

# Convertir las entradas del usuario a formato numérico
sexo_usuario = 1 if sexo_usuario == 'male' else 0
x_test = np.array([clase_usuario, edad_usuario, sexo_usuario])

# Calcular distancias y ordenarlas con QuickSort
distancias = distancia_euclidiana(X, x_test)

# Ordenar las distancias con QuickSort
quick_sort(distancias, 0, len(distancias) - 1)

# Predecir con KNN
K = 3
resultado = KNN(K, distancias, Y)

# Mostrar el resultado
if resultado == 1:
    print("La persona sobrevivió :)")
else:
    print("La persona murió X(") 