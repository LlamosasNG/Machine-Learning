import numpy as np

# Función de activación vista en clase
def activacion(z):
    if z >= 0.5:
        return 1
    else:
        return 0

# Función de entrenamiento
def entrenamiento():
    X1 = np.array([0, 0, 1, 1])
    X2 = np.array([0, 1, 0, 1])
    Yd = np.array([1, 1, 1, 0])
    l = 0.5  # Tasa de aprendizaje
    epocas = 10
    W = np.array([1, 0.2, 0.5])  # Pesos iniciales
    X0 = 1  # Entrada de sesgo
    bandera = 0

    for i in range(epocas):
        print(f"Época actual: {i + 1}")
        print(f"Pesos iniciales: {W[0]} , {W[1]}, {W[2]}")
        
        for j in range(len(X1)):
            # Cálculo de la salida del perceptrón
            z = X0 * W[0] + W[1] * X1[j] + W[2] * X2[j]
            Yobt = activacion(z)
            
            if Yobt == Yd[j]:
                bandera += 1
                print(f"Yd = {Yd[j]} y Yobt = {Yobt}")
            else:
                # Actualización de los pesos
                W[0] = W[0] - l * (Yobt - Yd[j]) * X0  # Peso asociado al sesgo
                W[1] = W[1] - l * (Yobt - Yd[j]) * X1[j]  # Peso asociado a X1
                W[2] = W[2] - l * (Yobt - Yd[j]) * X2[j]  # Peso asociado a X2
                bandera = 0
                print(f"Yd = {Yd[j]} y Yobt = {Yobt}")
                print(f"Pesos actualizados: {W[0]} , {W[1]}, {W[2]}")
                    
entrenamiento()
