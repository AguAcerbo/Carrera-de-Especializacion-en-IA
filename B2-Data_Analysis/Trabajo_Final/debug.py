
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

x_train['RainToday'] = x_train['RainToday'].replace({'No':0, 'Yes':1})
y_train = y_train.replace({'No':0, 'Yes':1})

x_test['RainToday'] = x_test['RainToday'].replace({'No':0, 'Yes':1})
y_test = y_test.replace({'No':0, 'Yes':1})

x_train['WindGustDir'] = x_train['WindGustDir'].replace({'E':(np.sin(0)+1)/2,
                                                         'ENE': (np.sin(22.5*np.pi/180)+1)/2,
                                                         'NE': (np.sin(45*np.pi/180)+1)/2,
                                                         'NNE':(np.sin(67.5*np.pi/180)+1)/2,
                                                         'N': (np.sin(90*np.pi/180)+1)/2,
                                                         'NNW': (np.sin(112.5*np.pi/180)+1)/2,
                                                         'NW': (np.sin(135*np.pi/180)+1)/2,
                                                         'WNW': (np.sin(157.5*np.pi/180)+1)/2,
                                                         'W': (np.sin(180*np.pi/180)+1)/2,
                                                         'WSW': (np.sin(202.5*np.pi/180)+1)/2,
                                                         'SW': (np.sin(225*np.pi/180)+1)/2,
                                                         'SSW': (np.sin(247.5*np.pi/180)+1)/2,
                                                         'S': (np.sin(270*np.pi/180)+1)/2,
                                                         'SSE': (np.sin(292.5*np.pi/180)+1)/2,
                                                         'SE': (np.sin(315*np.pi/180)+1)/2,
                                                         'ESE': (np.sin(337.5*np.pi/180)+1)/2})

x_train['WindDir9am'] = x_train['WindDir9am'].replace({'E':(np.sin(0)+1)/2,
                                                       'ENE': (np.sin(22.5)+1)/2,
                                                       'NE': (np.sin(45)+1)/2,
                                                       'NNE':(np.sin(67.5)+1)/2,
                                                       'N': (np.sin(90)+1)/2,
                                                       'NNW': (np.sin(112.5)+1)/2,
                                                       'NW': (np.sin(135)+1)/2,
                                                       'WNW': (np.sin(157.5)+1)/2,
                                                       'W': (np.sin(180)+1)/2,
                                                       'WSW': (np.sin(202.5)+1)/2,
                                                       'SW': (np.sin(225)+1)/2,
                                                       'SSW': (np.sin(247.5)+1)/2,
                                                       'S': (np.sin(270)+1)/2,
                                                       'SSE': (np.sin(292.5)+1)/2,
                                                       'SE': (np.sin(315)+1)/2,
                                                       'ESE': (np.sin(337.5)+1)/2})

x_train['WindDir3pm'] = x_train['WindDir3pm'].replace({'E':(np.sin(0)+1)/2,
                                                       'ENE': (np.sin(22.5)+1)/2,
                                                       'NE': (np.sin(45)+1)/2,
                                                       'NNE':(np.sin(67.5)+1)/2,
                                                       'N': (np.sin(90)+1)/2,
                                                       'NNW': (np.sin(112.5)+1)/2,
                                                       'NW': (np.sin(135)+1)/2,
                                                       'WNW': (np.sin(157.5)+1)/2,
                                                       'W': (np.sin(180)+1)/2,
                                                       'WSW': (np.sin(202.5)+1)/2,
                                                       'SW': (np.sin(225)+1)/2,
                                                       'SSW': (np.sin(247.5)+1)/2,
                                                       'S': (np.sin(270)+1)/2,
                                                       'SSE': (np.sin(292.5)+1)/2,
                                                       'SE': (np.sin(315)+1)/2,
                                                       'ESE': (np.sin(337.5)+1)/2})

x_test['WindGustDir'] = x_test['WindGustDir'].replace({'E':(np.sin(0)+1)/2,
                                                         'ENE': (np.sin(22.5)+1)/2,
                                                         'NE': (np.sin(45)+1)/2,
                                                         'NNE':(np.sin(67.5)+1)/2,
                                                         'N': (np.sin(90)+1)/2,
                                                         'NNW': (np.sin(112.5)+1)/2,
                                                         'NW': (np.sin(135)+1)/2,
                                                         'WNW': (np.sin(157.5)+1)/2,
                                                         'W': (np.sin(180)+1)/2,
                                                         'WSW': (np.sin(202.5)+1)/2,
                                                         'SW': (np.sin(225)+1)/2,
                                                         'SSW': (np.sin(247.5)+1)/2,
                                                         'S': (np.sin(270)+1)/2,
                                                         'SSE': (np.sin(292.5)+1)/2,
                                                         'SE': (np.sin(315)+1)/2,
                                                         'ESE': (np.sin(337.5)+1)/2})

x_test['WindDir9am'] = x_test['WindDir9am'].replace({'E':(np.sin(0)+1)/2,
                                                       'ENE': (np.sin(22.5)+1)/2,
                                                       'NE': (np.sin(45)+1)/2,
                                                       'NNE':(np.sin(67.5)+1)/2,
                                                       'N': (np.sin(90)+1)/2,
                                                       'NNW': (np.sin(112.5)+1)/2,
                                                       'NW': (np.sin(135)+1)/2,
                                                       'WNW': (np.sin(157.5)+1)/2,
                                                       'W': (np.sin(180)+1)/2,
                                                       'WSW': (np.sin(202.5)+1)/2,
                                                       'SW': (np.sin(225)+1)/2,
                                                       'SSW': (np.sin(247.5)+1)/2,
                                                       'S': (np.sin(270)+1)/2,
                                                       'SSE': (np.sin(292.5)+1)/2,
                                                       'SE': (np.sin(315)+1)/2,
                                                       'ESE': (np.sin(337.5)+1)/2})

x_test['WindDir3pm'] = x_test['WindDir3pm'].replace({'E':(np.sin(0)+1)/2,
                                                       'ENE': (np.sin(22.5)+1)/2,
                                                       'NE': (np.sin(45)+1)/2,
                                                       'NNE':(np.sin(67.5)+1)/2,
                                                       'N': (np.sin(90)+1)/2,
                                                       'NNW': (np.sin(112.5)+1)/2,
                                                       'NW': (np.sin(135)+1)/2,
                                                       'WNW': (np.sin(157.5)+1)/2,
                                                       'W': (np.sin(180)+1)/2,
                                                       'WSW': (np.sin(202.5)+1)/2,
                                                       'SW': (np.sin(225)+1)/2,
                                                       'SSW': (np.sin(247.5)+1)/2,
                                                       'S': (np.sin(270)+1)/2,
                                                       'SSE': (np.sin(292.5)+1)/2,
                                                       'SE': (np.sin(315)+1)/2,
                                                       'ESE': (np.sin(337.5)+1)/2})

# 'ENE': (np.sen(22.5)+1)/2,
# 'NE': (np.sen(45)+1)/2,
# 'NNE':(np.sen(67.5)+1)/2,
# 'N': (np.sen(90)+1)/2,
# 'NNW': (np.sen(112.5)+1)/2,
# 'NW': (np.sen(135)+1)/2,
# 'WNW': (np.sen(157.5)+1)/2,
# 'W': (np.sen(180)+1)/2,
# 'WSW': (np.sen(202.5)+1)/2,
# 'SW': (np.sen(225)+1)/2,
# 'SSW': (np.sen(247.5)+1)/2,
# 'S': (np.sen(270)+1)/2,
# 'SSE': (np.sen(292.5)+1)/2,
# 'SE': (np.sen(315)+1)/2,
# 'ESE': (np.sen(337.5)+1)/2


a = enumerate(x_train['Location'].unique())
city = {}
for code,ciudad in enumerate(x_train['Location'].unique()):
    city[ciudad] = code

x_test['Location'] = x_test['Location'].replace(city)
x_train['Location'] = x_train['Location'].replace(city)


features_with_nan = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'MinTemp', 'MaxTemp', \
                     'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', \
                     'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

data_cca = dataset.dropna(inplace=False)


# Pressure9am:	    Datos no NaN: 130395 	Datos Nan: 15065 	En%: 10.356799120033
# Pressure3pm:	    Datos no NaN: 130432 	Datos Nan: 15028 	En%: 10.331362573903478
# WindDir9am:	    Datos no NaN: 134894 	Datos Nan: 10566 	En%: 7.263852605527292
# WindGustDir:	    Datos no NaN: 135134 	Datos Nan: 10326 	En%: 7.09885879279527
# WindGustSpeed:	Datos no NaN: 135197 	Datos Nan: 10263 	En%: 7.055547916953114
# Humidity3pm:	    Datos no NaN: 140953 	Datos Nan: 4507 	En%: 3.09844630826344
# WindDir3pm:	    Datos no NaN: 141232 	Datos Nan: 4228 	En%: 2.906641000962464
# Temp3pm:	        Datos no NaN: 141851 	Datos Nan: 3609 	En%: 2.4810944589577892
# RainTomorrow:	    Datos no NaN: 142193 	Datos Nan: 3267 	En%: 2.245978275814657
# RainToday:	    Datos no NaN: 142199 	Datos Nan: 3261 	En%: 2.241853430496356
# Rainfall:	        Datos no NaN: 142199 	Datos Nan: 3261 	En%: 2.241853430496356
# WindSpeed3pm:	    Datos no NaN: 142398 	Datos Nan: 3062 	En%: 2.105046060772721
# Humidity9am:	    Datos no NaN: 142806 	Datos Nan: 2654 	En%: 1.8245565791282827
# WindSpeed9am:	    Datos no NaN: 143693 	Datos Nan: 1767 	En%: 1.214766946239516
# Temp9am:	        Datos no NaN: 143693 	Datos Nan: 1767 	En%: 1.214766946239516
# MinTemp:	        Datos no NaN: 143975 	Datos Nan: 1485 	En%: 1.0208992162793895
# MaxTemp:	        Datos no NaN: 144199 	Datos Nan: 1261 	En%: 0.8669049910628351


x_train['MinTemp']=x_train['MinTemp'].fillna(x_train['MinTemp'].median())

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)
x_train_knn_imp = imputer.fit_transform(x_train)

x_train.iloc[:,'Pressure9am'] = x_train_knn_imp[:,15]



df_train_c2 = pd.DataFrame(np.concatenate([x_train_c2,y_train_c2[:,np.newaxis]], axis=1), 
                        columns=data_c2.columns[np.concatenate([sel.get_support(), [True]])])


r = df_train.corr(method='pearson')
MI = mutual_info_regression(x_train_cca, y_train_cca)

fig, ax = plt.subplots(2,1, figsize=(22,15))
ax[0].set_title('Información mutua')
sns.heatmap([MI],ax=ax[0],cmap=sns.diverging_palette(220,10,as_cmap=True), annot=True,fmt=".2f")
ax[0].set_xticklabels(df_train.columns.values[:-1])

ax[1].set_title('Correlación de Pearson')
sns.heatmap([r.iloc[-1][:-1]],cmap=sns.diverging_palette(220,10,as_cmap=True),ax=ax[1], annot=True,fmt=".2f")
ax[1].set_xticklabels(df_train.columns.values[:-1])

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    
X_train_mi, X_test_mi, mi = select_features_mutual_info(x_train_red, y_train_cca, x_train_red)
X_train_pc, X_test_pc, pc = select_features_pearson(x_test_red, y_test_cca, x_test_red)
fig,axes = plt.subplots(2,1,figsize=(16,15))
axes[0].set_title('Información mutua')
sns.barplot( x = [c for c in df_train.columns[:-1] ], y = mi.scores_,ax=axes[0])
axes[1].set_title('Pearson')
sns.barplot( x = [c for c in df_train.columns[:-1] ], y = pc.scores_,ax=axes[1]); #Grafico el F score de cada feature
pc.scores_
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=22)

