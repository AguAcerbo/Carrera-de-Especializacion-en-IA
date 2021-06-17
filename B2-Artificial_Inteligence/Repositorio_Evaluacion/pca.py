import numpy as np


def pca_numpy(x, n_components=2):
    # centro el dataset
    x_cent = x.copy()
    x_cent = x_cent - x.mean(axis=0)

    # Obtengo la matriz de covarianza y calculo sus autovectores y autovalores:
    s = (x_cent.T @ x_cent) / x_cent.shape[0]
    autovalores, autovectores = np.linalg.eig(s)

    # Ordena los autovectores en el sentido de los autovalores decrecientes
    arg_sort = np.argsort(autovalores)[::-1]
    autovalores = autovalores[arg_sort]
    autovectores = autovectores[:, arg_sort]

    # Selecciono los n_components de mayor valor. (n_components <= m)
    b = autovectores[:, :n_components]

    z = np.dot(x_cent, b)
    x_recuperada = np.dot(z, b.T) + x.mean(axis=0)

    return b, z, x_recuperada


# Codigo de verificacion:
# x = np.array([[0.8,0.7],[0.1, -0.1]])
#
# B, z, x_recuperada = PCA_numpy(x,1)
# print("PCA en Numpy")
# print("Matriz de transformacion\n",B)
# print("Matriz reducida\n",z)
# print("Matriz recuperada\n", x_recuperada)
#
# from sklearn.decomposition import PCA
# print("\nPCA por SCIKIT")
# pca = PCA(n_components=1)
# pca.fit(x)
# print("Matriz reducida\n",pca.transform(x))
# print("Matriz recuperada\n",pca.inverse_transform(pca.transform(x)))
