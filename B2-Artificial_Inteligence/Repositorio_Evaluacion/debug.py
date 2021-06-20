# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:46:15 2021

@author: Agustin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from split_dataset import dataset_manipulation


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)
        self.x = None
        self.y = None
        
    @staticmethod
    def _build_dataset( path):
        structure = [('x', float),
                     ('y', float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]))
                        for i, line in enumerate(data_csv) if i != 0)
            data = np.fromiter(data_gen, structure)
        
        return data

    def split(self, percentage):  # 0.8
        self.x = self.dataset['x']
        self.y = self.dataset['y']

        # X.shape[0] -> 10 (filas)

        permuted_idxs = np.random.permutation(self.x.shape[0])
        # 2,1,3,4,6,7,8,5,9,0

        train_idxs = permuted_idxs[0:int(percentage * self.x.shape[0])]
        # permuted_idxs[0:8]
        # [2,1,3,4,5,6,7,8,5]

        test_idxs = permuted_idxs[int(percentage * self.x.shape[0]): self.x.shape[0]]
        # [9,0]

        x_train = self.x[train_idxs]
        x_test = self.x[test_idxs]

        y_train = self.y[train_idxs]
        y_test = self.y[test_idxs]

        return x_train, x_test, y_train, y_test

def mse(target, prediction):
    return np.sum((target - prediction) ** 2) / target.size

def mini_batch_gradient_descent(x_train_mb, y_train_mb, x_test_mb, y_test_mb, lr=0.1, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 5
    n = x_train_mb.shape[0]
    m = x_train_mb.shape[1]

    # initialize random weights
    w = np.random.randn(m).reshape(m, 1)
    
    mse_train = []
    mse_val = []
    for epoch in range(amt_epochs):
        idx = np.random.permutation(x_train_mb.shape[0])
        x_train_mb = x_train_mb[idx]
        y_train_mb = y_train_mb[idx]

        batch_size = int(len(x_train_mb) / b)
        for i in range(0, len(x_train_mb), batch_size):
            end = i + batch_size if i + batch_size <= len(x_train_mb) else len(x_train_mb)
            batch_x = x_train_mb[i: end]
            batch_y = y_train_mb[i: end]
            
            batch_y = batch_y.reshape(-1,1) #Con esta linea corrijo el error que tenia debido a las dimensiones de las matrices.
            
            prediction = np.matmul(batch_x, w)  # nx1
            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_x, axis=0)
            grad_mul = -2/batch_size * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            w = w - (lr * gradient)
        
        mse_train.append(mse(y_train_mb.reshape(-1,1), np.matmul(x_train_mb, w)))
        mse_val.append(mse(y_test_mb.reshape(-1,1), np.matmul(x_test_mb, w)))

    return w, mse_train, mse_val

dataset=Data(r'clase_8_dataset.csv')
x_train, x_test, y_train, y_test = dataset.split(0.8)
x_n3 = dataset_manipulation(x_train/np.max(x_train), polinomy_grade=3, bias=True)   

x_train_mb = x_n3.copy()
y_train_mb = y_train.copy()

x_n3_test = dataset_manipulation(x_test/np.max(x_train), polinomy_grade=3, bias=True)
amt_epochs=100
w, mse_train, mse_val = mini_batch_gradient_descent(x_train_mb, y_train_mb, x_n3_test, y_test, lr=0.1, amt_epochs=100)

e = np.arange(amt_epochs)

plt.figure(figsize=(20,10))
plt.subplot(1, 1, 1)
#plt.gca().set_title('Error')
plt.plot(e, mse_train, '--')
plt.plot(e, mse_val, '-')
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend(['Error de entrenamiento', 'Error de validacion'])

