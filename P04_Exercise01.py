import pandas as pd

# Cargar los datos del archivo CSV
ruta_archivo = './train.csv'
datos = pd.read_csv(ruta_archivo)

# Asume que 'etiqueta' es la columna objetivo
Y = datos['Survived'].tolist()
Sex = datos['Sex'].tolist()
Age = datos['Age'].tolist()
Class = datos['Pclass'].tolist()

# Eliminar todas las columnas innecesarias de una sola vez para definir X
X = datos.drop(columns=['Survived', 'Sex', 'Age', 'Pclass'])
