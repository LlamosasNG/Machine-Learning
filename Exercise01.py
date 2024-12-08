import matplotlib.pyplot as plt
import random
import numpy as np

def distancia_ecluidiana(X,x_test):
  ## X puede un punto o un vector de dos posiciones
  ## x_test es un solo punto a clasificar
  x_test = np.array(x_test)
  d = (X-x_test)**2
  distancia = [None] * len(d)
  for i in range(len(d)):
      distancia[i]= np.sqrt(d[i][0]+d[i][1])
  return distancia

def Kmeans(data, k, epocas):
  data=np.array(data) #iguales
  numero_datos=len(data) #numero_datos lo tiene el profe
  indices_usados = [-1]*k #iguales
  centroides = [] #igaules
  numero_datos= len(data) #iguales
  num_datos, num_caracteristicas=data.shape #aqui pone num_datos
  ## Asignar aleatoriamente los centroides 
  for i in range (k):
    while True:
      random_centroides = random.randint(0, numero_datos -1) #sacar un numero aleatorio con distribucion especifica de tipo int
      duplicado = False
      for j in range(i):
        if indices_usados[j] == random_centroides:
          duplicado = True
          break
      if duplicado == False:
        indices_usados[i] = random_centroides
        centroides = centroides + [data[random_centroides]]
        break
        
############################################################################
  # Calcular la distancia de los puntos a los centroides
  for iteraciones in range(epocas):
      clouster_asignados = [0]*num_datos
      for i in range(num_datos):
          distancias = [0]*k
          for j in range(k):
              distancias[j] = distancia_ecluidiana([centroides[j]], data[i])[0] #Distancia del punto a los 4 centros
          min_distancia_index=np.argmin(distancias)
          clouster_asignados[i] = min_distancia_index
        
########## Hacer el prodemio de los datos que pertenecen a una clase (clouster) para obtener nuevos centroides
  Nuevos_Centroides=[]
  for closuter_index in range(k): #recorre todas las clases clouster 1, 2 ,3 hasta clase k
    puntos_en_clouster=[]
    for los_datos in range(num_datos):
      if clouster_asignados[los_datos]==closuter_index:
        puntos_en_clouster += [data[los_datos]] #apila los datos de la clase en una lista
        
    ######## Optencion de los promedios de los datos de la clase
    if len(puntos_en_clouster)>0:
      clouster_sumas= [0]* num_caracteristicas
      for dato_en_clouster in puntos_en_clouster: #recorre todos los puntos en el clouster
        for caracteristicas in range(num_caracteristicas): #recorre todas las caracteristicas
          clouster_sumas[caracteristicas] += dato_en_clouster[caracteristicas] #suma todos los valores de la primera y segunda columna
      promedio_clouster = [clouster_sumas[j]/len(puntos_en_clouster) for j in range (num_caracteristicas)]
      Nuevos_Centroides += [promedio_clouster]
    else:
      Nuevos_Centroides += [centroides[closuter_index]]
      
  ## Early stop
    #print(f"Centroides actuales: {len(centroides)}, Nuevos centroides: {len(Nuevos_Centroides)}")
    variacion = True
    epsilon=0.000001
    for i in range(k):
      for j in range(num_caracteristicas):
        if(abs(Nuevos_Centroides[i][j]-centroides[i][j])) > epsilon:
          variacion= False
          break
      if variacion==False:
        break
      if variacion == True:
        break
    
      centroides = Nuevos_Centroides

  ## Agrupamiento
  clousters= [[] for _ in range(k)]
  for i in range(num_datos):
    clousters[clouster_asignados[i]] = clousters[clouster_asignados[i]] + [data[i]] 
  
  return centroides,clousters


def graficar_puntos(data):
  for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], color='red')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Kmeans")
  plt.grid()
  plt.show()

data = [[1,2],[1.5,2.3],[1.2,1.9],
        [4,5],[4.1,5.1],[4.4,5.3], 
        [9,10],[9.1,10.1],[8.8,10.4],
        [15,16.1],[15.2,16.5],[14.9,15.9]
]

k= 6
epocas=100
print(Kmeans(data, k, epocas))
graficar_puntos(data)