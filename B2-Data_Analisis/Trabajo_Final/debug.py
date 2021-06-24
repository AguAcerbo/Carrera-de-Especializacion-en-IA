
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

from sklearn.model_selection import train_test_split

data_weather = dataset.dropna(subset=['RainTomorrow'], inplace=False)

data_weatherAUS = data_weather.copy()
#data_weatherAUS['day'] = pd.DatetimeIndex(data_weather['Date']).day
data_weatherAUS['Month'] = pd.DatetimeIndex(data_weather['Date']).month
#data_weatherAUS['year'] = pd.DatetimeIndex(data_weather['Date']).year

data_weatherAUS.drop(['Date'], axis=1,inplace=True)

features = ['Month', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', \
            'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', \
            'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', \
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm' , \
            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']

x_train, x_test, y_train, y_test = train_test_split(
    data_weatherAUS[features], 
    data_weatherAUS['RainTomorrow'],  
    test_size=0.3,
    random_state=42)

Pressure9am:	Datos NaN: 9825	En%: 9.870899683528407
Pressure3pm:	Datos NaN: 9802	En%: 9.847792233887578
WindDir9am:	Datos NaN: 7018	En%: 7.05078615562365
WindGustDir:	Datos NaN: 6489	En%: 6.519314813884563
WindGustSpeed:	Datos NaN: 6442	En%: 6.472095242879389
WindDir3pm:	Datos NaN: 2634	En%: 2.646305319736776
Humidity3pm:	Datos NaN: 2543	En%: 2.554880192896971
Temp3pm:	Datos NaN: 1921	En%: 1.9299743808710503
WindSpeed3pm:	Datos NaN: 1833	En%: 1.8415632692017883
Humidity9am:	Datos NaN: 1231	En%: 1.2367508916461547
Rainfall:	Datos NaN: 1034	En%: 1.0388305621138294
RainToday:	Datos NaN: 1034	En%: 1.0388305621138294
WindSpeed9am:	Datos NaN: 926	En%: 0.9303260159742804
Temp9am:	Datos NaN: 627	En%: 0.6299291706434922
MinTemp:	Datos NaN: 460	En%: 0.46214899281659716
MaxTemp:	Datos NaN: 235	En%: 0.2360978550258703