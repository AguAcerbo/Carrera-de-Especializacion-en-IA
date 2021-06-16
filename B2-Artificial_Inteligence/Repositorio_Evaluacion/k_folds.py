import numpy as np
from metrics import mse
from lineal_polinomic_regresion import ConstantModel, LinearRegression, LinearRegressionWithB


def k_folds(x_train, y_train, k=10):
    """
    Establece un loop para los k-folds, por defecto se definen 10.
    Devuelve media del error cuadratico obtenido en cada loop

    1ro
     ___ ___ ___       ___ ___
    | 1 | 2 | 3 |     |k-1| k |
    |Val|Tra|Tra| ... |Tra|Tra|

    2do
     ___ ___ ___       ___ ___
    | 1 | 2 | 3 |     |k-1| k |
    |Tra|Val|Tra| ... |Tra|Tra|

    k-esimo
     ___ ___ ___       ___ ___
    | 1 | 2 | 3 |     |k-1| k |
    |Tra|Tra|Tra| ... |Tra|Val|


    """

    l_regression = LinearRegression()

    chunk_size = int(len(x_train) / k)
    mse_list = []
    for i in range(0, len(x_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(x_train) else len(x_train)
        new_x_valid = x_train[i: end]
        new_y_valid = y_train[i: end]
        new_x_train = np.concatenate([x_train[: i], x_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_x_train, new_y_train)
        l_regression.predict(new_x_valid)
        mse_list.append(mse(new_y_valid, l_regression.predicted))

    mean_mse = np.mean(mse_list)

    return mean_mse
