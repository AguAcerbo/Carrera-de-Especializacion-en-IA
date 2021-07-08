# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:45:23 2021

@author: Agustin
"""
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np


cats_train = tf.keras.preprocessing.image.load_img(r'Documents\VpC2_Datasets\Clase2\perros_y_gatos\train\cats')