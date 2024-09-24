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
p = 0
p = vp / (vp + fp)

# Cálculo de la exactitud 
ex = 0
ex = (vp + vn) / ( vp + vn + fp + fn)

# Cálculo de la recall
r = 0
r = vp / (vp + fn)

# Cálculo de la presición
F2 = 0
F2 = 2 * p * r / ( p + r)

print(p)
print(ex)
print(r)
print(F2)

# for i in range(filas):
    #for j in range(columnas):
        #print(X[i,j])
        