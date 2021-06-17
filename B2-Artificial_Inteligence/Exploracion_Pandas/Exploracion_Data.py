# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:51:18 2021

@author: Agustin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("data/car data.csv")

data.head() # visualizo features
data.info() # tipo de datos de features
data.describe() # info estadistica
data.isnull().sum() # veo si tiene valores nulos

# Exploratory Data Analysis
sns.barplot(x='x1',y='Y',data=data,palette='spring') # relacion entre feature x1 y salida

plt.figure(figsize=(10,10)) # relacion entre feature x2 y salida
sns.lmplot(x='x2',y='Y',data=data)

# Estudio de Correlación
corr_matrix = data.corr(method='pearson')
# Heatmap matriz de correlaciones
# ============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 18))
sns.heatmap(
    corr_matrix,
    square    = True,
    ax        = ax
)
ax.tick_params(labelsize = 10)

sns.pairplot(data)

# Feature Engineering

# Vamos a crear una nueva feature que en vez de almacenar el año del modelo,
# almacene los años transcurridos al presente. De este modo tiene una relación 
# más directa con el precio de venta actual.
data['vehicle_age'] = 2021- data['Year']
data.head()
data.drop(columns=['Year'],inplace=True)

# Vamos a codificar las variables categóricas usando get_dummies().
# Esto generalmente trae inconvenientes en los modelos lineales.
data=pd.get_dummies(data,columns=['Fuel_Type','Transmission','Seller_Type'], drop_first=True)

data.head()