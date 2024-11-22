from re import S
import pandas as pd
import numpy as np

# 1.- Cargar los datos del archivo CSV
ruta_archivo = './train.csv'
datos = pd.read_csv(ruta_archivo)

# Asume que 'etiqueta' es la columna objetivo
Y = datos['Survived'].tolist()
Sex = datos['Sex'].tolist()
Age = datos['Age'].tolist()
Class = datos['Pclass'].tolist()

# 2.- Eliminar todas las columnas innecesarias de una sola vez para definir X
X = datos.drop(columns=['Survived', 'Sex', 'Age', 'Pclass'])

# 3.- Realizar la medición de la moda y rellenar espacios vacíos para los vectores de características
moda = 'male' if Sex.count('male') >= Sex.count('female') else 'female'
Sex = [moda if s == '' else s for s in Sex]

# Convertir la lista Age a un arreglo de numpy, reemplazando valores no numéricos con NaN
Age_array = np.array([float(a) if a != '' and a != 'NaN' else np.nan for a in Age])
# Calcular el promedio ignorando valores NaN
promedio = int(np.nanmean(Age_array))  # np.nanmean ignora valores NaN
# Reemplazar valores NaN con el promedio
Age_array = np.where(np.isnan(Age_array), promedio, Age_array).astype(int)  # Reemplazar NaN y convertir a enteros
# Convertir el arreglo numpy de nuevo a una lista (opcional)
Age = Age_array.tolist()
# Reemplazar espacios vacíos con el promedio
Age = [promedio if a == 'NaN' else float(a) for a in Age]

# Clase
conteo_1 = sum(1 for c in Class if c == '1')
conteo_2 = sum(1 for c in Class if c == '2')
conteo_3 = sum(1 for c in Class if c == '3')

if conteo_1 >= conteo_2 and conteo_1 >= conteo_3:
    moda = '1'
elif conteo_2 >= conteo_1 and conteo_2 >= conteo_3:
    moda = '2'
else:
    moda = '3'

Class = [moda if c == '' else c for c in Class]

# División: 80% para entrenamiento, 20% para prueba
Size = int(len(Y) * 0.8) 
Sex_train = Sex[:Size]  
Sex_test = Sex[Size:]

Age_train = Age[:Size ]  
Age_test = Age[Size:]

Class_train = Class[:Size]  
Class_test = Class[Size:]  

