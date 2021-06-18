import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

# Creamos los datos con NumPy. Pueden probar con diferentes funciones no lineales o incluso combinaciones de senos.
# Se puede probar también modificando el desvío estandar del ruido agregado.

# Generamos los 'x' de nuestra función no lineal a emplear
x = np.arange(60, 300, 4) # vector en grados
x = x*np.pi/180 # conversión a radianes
np.random.seed(10)

# Calculamos la función y agregamos ruido
stdv = 0.15
y = np.sin(x) + np.random.normal(0, stdv, len(x))

# convertimos los datos en un dataframe de pandas
plt.plot(x,y,'.')

# Pueden crear las features polinómicas manualmente con NumPy empleando np.power o con SKlearn (PolynomialFeatures)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(15, include_bias=False)
poly_data = poly.fit_transform(x.reshape(-1, 1))

# Creamos los column names para el dataframe (va a ser más fácil dar seguimiento luego)
colname = ['x']
for i in range(2, 16):
    colname.append('x_%d'%i)

colname.append('y')

# Creamos el dataframe
data = pd.DataFrame(np.column_stack([poly_data,y]),columns=colname)
data.head()
