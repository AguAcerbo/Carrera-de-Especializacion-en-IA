
"""
Created on Wed Jun 23 00:29:26 2021

@author: Agustin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns; sns.set()

dataset = pd.read_csv("weatherAUS.csv")



dataset.loc['Date']



    
    
# - Variables de entrada:
#   - Realizar los siguientes análisis por tipo de variable.
#     - Numéricas: 
#       - Obtener conclusiones acerca de la distribución de los datos.
#     - Categóricas
#         - Obtener conclusiones acerca de cardinalidad, representación de cada categoría, etc.
#     - Compuestas/otros. ¿Cómo pueden tratarse para utilizarlas en el problema elegido?