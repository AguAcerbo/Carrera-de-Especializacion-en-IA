# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:00:42 2021

@author: Agustin
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

def kfold_pipeline(models_pipeline, samples , target, x_test, y_test, k = 5):
    
    """
    Esta funcion recibe:
        models_pipeline: el pipeline de los modelos a evaluar
        samples: set de entrenamiento
        target: salidas del set de entrenamiento
        x_test: set de testeo
        y_test: set de testeo
        k: el numero de segmentos que utilizara el algoritmo de K-Fold
    """
    results = {}
    for model in models_pipeline:
        model_name = model[0]
        results[model_name] = {}
        
        train_auc = 0
        valid_auc = 0
        
        kf = KFold(n_splits=k)
        kf.get_n_splits(samples)
        
        for train_index, valid_index in kf.split(samples, target):
            x_valid = samples.iloc[valid_index]
            y_valid = target.iloc[valid_index]
            x_train = samples.iloc[train_index]
            y_train = target.iloc[train_index]
            
            model[1].fit(x_train, y_train)
            pred_train = model[1].predict_proba(x_train)
            pred_valid = model[1].predict_proba(x_valid)
            
            if len(pred_train.shape)>1 and (pred_train.shape[1]>1):
                train_auc = train_auc + roc_auc_score(y_train, pred_train[:,1]) / k
                valid_auc = valid_auc + roc_auc_score(y_valid, pred_valid[:,1]) / k
            else:
                train_auc = train_auc + roc_auc_score(y_train, pred_train) / k
                valid_auc = valid_auc + roc_auc_score(y_valid, pred_valid) / k
        
        pred_test = model[1].predict_proba(x_test)

        results[model_name]["Train"] = train_auc
        results[model_name]["Valid"] = valid_auc
        results[model_name]["test"] = roc_auc_score(y_test, pred_test[:,1])

    models_results = pd.DataFrame(results).T
    return models_results