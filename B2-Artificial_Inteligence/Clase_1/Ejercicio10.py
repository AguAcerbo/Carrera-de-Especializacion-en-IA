# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:32:39 2021

@author: Agustin
"""

import numpy as np
import pandas as pd
import pickle as pkl
import os

class Rating_Dataset(object):
    
    instance = None
    dataset = None
    
    def __new__(cls, path):
        if Rating_Dataset.instance is None:
            Rating_Dataset.instance=super(Rating_Dataset,cls).__new__(cls)
            return Rating_Dataset.instance
        else:
            return Rating_Dataset.instance
    
    def __init__(self, path):
        if os.path.isfile(path+'/ratings.pkl'):
            
            with open(path+'/ratings.pkl','rb') as file:
                self.dataset=pkl.load(file)
            
        else:
            structure = [('userID', np.uint32),
                         ('movieID', np.uint32),
                         ('rating', np.float32),
                         ('timestamp', np.uint32)]
            data=pd.read_csv(path+'/ratings.csv', delimiter=',')
            s = data.dtypes
            res2 = np.array([tuple(x) for x in data.values], \
                            dtype=list(zip(s.index, s)))
                
            self.dataset=np.array(res2,dtype=structure)
            
            with open(path+'/ratings.pkl','wb') as file:
                pkl.dump(self.dataset,file,protocol=pkl.HIGHEST_PROTOCOL)
                
            pass
    
    def get_dataset(self):
        return self.dataset
    
rating = Rating_Dataset(r"C:\Users\Agustin\Documents\RepositorioCEIA_Git\Carrera-de-Especializacion-en-IA\B2-Artificial_Inteligence\Clase_1\archive")
print(rating.get_dataset()[:5])
