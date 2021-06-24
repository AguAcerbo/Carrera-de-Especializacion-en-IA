# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 22:29:16 2021

@author: Agustin
"""
import pandas as pd

def set_data_weatherAUS(dt):
    
    # Se mantiene el mes de la muestra borrando la var. compuesta de year-month-day
    data_weatherAUS = dt.copy()
    #data_weatherAUS['Day'] = pd.DatetimeIndex(dt['Date']).day
    data_weatherAUS['Month'] = pd.DatetimeIndex(dt['Date']).month
    #data_weatherAUS['Year'] = pd.DatetimeIndex(dt['Date']).year
    data_weatherAUS.drop(['Date'], axis=1,inplace=True)
    
    # One-Hot Encoding en variables categoricas de 'RainToday' y 'RainTomorrow'
    data_weatherAUS['RainToday'] = data_weatherAUS['RainToday'].replace({'No':0, 'Yes':1})