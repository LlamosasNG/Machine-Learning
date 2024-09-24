import numpy as np

# Vector de características
X = np.array([[0,0,0,1,0,1], 
              [1,0,0,0,0,0],
              [1,0,1,0,1,0],
              [1,1,0,0,1,1],
              [1,1,1,0,0,0],
              [1,1,1,1,0,0],
              [1,1,1,1,1,0],
              [1,1,1,1,1,1]])

# print(np.shape(X))
# print(len(X))

# Primero definimos la salida esperada
Yd = np.array([1,0,1,0,1,1,0,0])

# Definir Y(obt) solamente para {este caso}
Yobt = np.array([1,0,0,1,1,1,0,0])

filas = np.shape(X)[0]
columnas = np.shape(X)[1]

# Obtenemos longitud de los vectores y para corraborar que sus dimenciones coincidan
tamanioYd = len(Yd)
tamanioYobt = len(Yobt)

# Obtuvimos la cantidad de verdaderos positivos, verdaderos negativos, falsos negativos y falsos positivos.
vp=0
vn=0
fn=0
fp=0

if tamanioYd == tamanioYobt:
    for i in range(tamanioYd):
        if Yd[i] == 1 and Yobt[i] == 1:
            vp = vp+1
        elif Yd[i] == 0 and Yobt[i] == 0:
            vn = vn+1
        elif Yd[i] == 1 and Yobt[i] == 0:
            fn = fn+1
        elif Yd[i] == 0 and Yobt[i] == 1:
            fp = fp+1

# Cálculo de la presición
P = 0
P = vp / (vp + fp)

# Cálculo de la exactitud 
Ex = 0
Ex = (vp + vn) / ( vp + vn + fp + fn)

# Cálculo del recall
R = 0
R = vp / (vp + fn)

# Cálculo del F2 Score
F2 = 0
F2 = 2 * P * R / ( P + R)

print(f"Presicion: { P }")
print(f"Exactitud: { Ex }")
print(f"Recall: { R }")
print(f"F2 Score: { F2 }")

# for i in range(filas):
    #for j in range(columnas):
        #print(X[i,j])
        