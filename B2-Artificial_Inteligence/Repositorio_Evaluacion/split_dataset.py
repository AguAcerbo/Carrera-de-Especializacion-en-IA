import numpy as np


def train_val_test(dataset, train_percent=70, val_percent=20):
    """Separa el dataset en training dataset, validation dataset y test dataset
    con proporciones por defecto 70%/20%/10% respectivamente """
    dataset_index = np.random.permutation(dataset.shape[0])

    train_data_idx = np.uint(np.around(dataset.shape[0] * train_percent / 100))
    val_data_idx = np.uint(np.around(dataset.shape[0] * val_percent / 100))
    test_data_idx = np.uint(dataset.shape[0]) - train_data_idx - val_data_idx

    train_data = dataset[dataset_index[0:train_data_idx]]
    val_data = dataset[dataset_index[train_data_idx:train_data_idx + val_data_idx]]
    test_data = dataset[dataset_index[train_data_idx + val_data_idx:train_data_idx + val_data_idx + test_data_idx]]

    return train_data, val_data, test_data


def dataset_polinomic(x, polinomy_grade=1):
    dataset_x = x.copy()
    for grade in range(2, polinomy_grade+1):
        dataset_x = np.vstack((np.power(x, grade), dataset_x))
    return dataset_x.T
