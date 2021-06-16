import numpy as np


def norma_0(matriz):
    """Calcula la norma cero de los vectores filas de la matriz
    que recibe. Es decir, devuelve la cantidad de valores distintos
    de cero que hay en cada vector"""
    return np.count_nonzero(matriz, axis=1)


def norma_p(matriz, p=1, ax=1):
    """Calcula la norma p de los vectores
    filas de la matriz que recibe."""
    return np.power(np.sum(np.power(np.abs(matriz), p), axis=ax), 1/p)


def normainf(matriz):
    """Calcula la norma infinito de los vectores filas de la matriz
    que recibe. Es decir, devuelve el elemento de mayor valor que
    hay en cada vector"""
    return np.amax(matriz, axis=1)
