import numpy as np
from norms import norma_p
import matplotlib.pyplot as plt


def k_means(x, n, iteraciones):
    """ Recibe:
        * x: matriz que representa una nube de puntos, cada fila es un punto.
        * n: numero de clusters a establecer
        * iteraciones: numero de ciclos para la optimizacion de los centroides de los clusters

        Esta funcion define n centroides de manera aleatoria.

        En cada iteracion se calcula la distancia de cada punto a los centroides. Luego redefine cada coordenada
        del centroide como el promedio de los puntos que estan mas proximos a el.

        Se devuleven los paramentros:
        * Centroides originales
        * Centroides optimizados
        * El nro de cluster al que pertenece cada punto en x.
        * Funcion Costo Original
        * Funcion Costo Final
        """

    # Defino n puntos aleatorios como centroides para mi estudio
    # centroides_original = np.random.randint(np.min(x), np.max(x), (n, x.shape[1]))

    # Establezco los centroides originales como una eleccion aleatorias de n puntos de x.
    # De esta manera disminuye la posibilidad de alcanzar un minimo local en la clusterizacion.
    centroides_original = x[np.random.randint(np.min(x), np.max(x), n)].copy()

    centroides = centroides_original.copy()
    for i in range(iteraciones):
        centroides_exp = centroides[:, np.newaxis]
        dist = norma_p(centroides_exp - x, p=2, ax=2)
        arg_min = np.argmin(dist, axis=0)

        for centroid in range(centroides.shape[0]):
            centroides[centroid] = np.mean(x[arg_min == centroid, :], axis=0)

        if i == 0:
            dist0 = 0
            for cluster in range(n):
                dist0 = dist0 + np.sum(norma_p(centroides[cluster] - x[arg_min == cluster], p=2, ax=1))
            func_costo_0 = dist0 / x.shape[0]

    distf = 0
    for cluster in range(n):
        print(
            f"Centroide original {cluster + 1}: {centroides_original[cluster, :]}"
            f"\tCentroide Optimizado {cluster + 1}: {centroides[cluster, :]}")

        distf = distf + np.sum(norma_p(centroides[cluster] - x[arg_min == cluster], p=2, ax=1))
        func_costo_f = distf / x.shape[0]

    print(f"Funcion Costo Inicial:{func_costo_0}\tFuncion Costo Final:{func_costo_f}")

    return centroides_original, centroides, arg_min, func_costo_0, func_costo_f


# *******************************************************************************************************

# cantidad de centroides
n = 3
# Nube de puntos
x = np.random.randint(-20, 20, (200, 2))
# nro de iteraciones para algoritmo K-means
iteraciones = 5

centroides_original, centroides, cluster_id, J0, Jf = k_means(x, n, iteraciones)

if n == 2:
    x0 = x[cluster_id == 0].copy()
    x1 = x[cluster_id == 1].copy()

    plt.figure(figsize=(10, 10))
    plt.plot(x0[:, 0], x0[:, 1], 'o', markersize=2, c='red')
    plt.plot(x1[:, 0], x1[:, 1], 'o', markersize=2, c='blue')
    plt.plot(centroides_original[:, 0], centroides_original[:, 1], 'x', markersize=10)
    plt.plot(centroides[:, 0], centroides[:, 1], '^', markersize=10)
    plt.show()
elif n == 3:
    x0 = x[cluster_id == 0].copy()
    x1 = x[cluster_id == 1].copy()
    x2 = x[cluster_id == 2].copy()

    plt.figure(figsize=(10, 10))
    plt.plot(x0[:, 0], x0[:, 1], 'o', markersize=2, c='red')
    plt.plot(x1[:, 0], x1[:, 1], 'o', markersize=2, c='blue')
    plt.plot(x2[:, 0], x2[:, 1], 'o', markersize=2, c='green')
    plt.plot(centroides_original[:, 0], centroides_original[:, 1], 'x', markersize=10)
    plt.plot(centroides[:, 0], centroides[:, 1], '^', markersize=10)
    plt.show()
