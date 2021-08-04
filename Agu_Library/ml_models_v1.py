# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:54:30 2021

@author: Agustin
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Evaluación de modelos de prueba
def pipeline_ml_models(preprocessing_pipeline):
    
    # Modelos a ensayar.
    models = [
        ("RandomForest", 
             Pipeline([
                 ('preprocessor', preprocessing_pipeline),
                 ('model', RandomForestClassifier(n_estimators=200, random_state=42)) 
            ])
        ),
      
        ("Adaboost", 
             Pipeline([
                 ('preprocessor', preprocessing_pipeline),
                 ('model', AdaBoostClassifier(n_estimators=200, random_state=42)) 
            ])
        ),
      
        ("LogisticRegression", 
             Pipeline([
                 ('preprocessor', preprocessing_pipeline),
                 ('model', LogisticRegression(random_state=42, solver='lbfgs') )
             ])
        ),
      
        ("KNN", 
             Pipeline([
                 ('preprocessor', preprocessing_pipeline),
                 ('model', KNeighborsClassifier(n_neighbors=5) )
            ])
        ),
      
        ("SVM", 
             Pipeline([
                 ('preprocessor', preprocessing_pipeline),
                 ('model', SVC(random_state=44, probability=True, gamma='auto') )
            ])
        )
    ]
    return models