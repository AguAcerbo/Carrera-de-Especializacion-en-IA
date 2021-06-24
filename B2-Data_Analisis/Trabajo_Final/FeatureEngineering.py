# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 22:29:16 2021

@author: Agustin
"""

def set_data_weatherAUS(dt):
    data_weather = dt.dropna(subset=['RainTomorrow'], inplace=False)
