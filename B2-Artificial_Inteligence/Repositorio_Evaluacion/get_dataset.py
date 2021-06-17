import numpy as np


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    @staticmethod
    def _build_dataset(path):
        structure = [('feature1', float),
                     ('salida', float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[1]), float(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i != 0)
            data = np.fromiter(data_gen, structure)

        return data

    def split(self, train_percent=70, val_percent=20):  # 0.8
        x = self.dataset['feature1']
        y = self.dataset['salida']

        dataset_index = np.random.permutation(x.shape[0])

        train_idx = np.uint(np.around(x.shape[0] * train_percent / 100))
        val_idx = np.uint(np.around(x.shape[0] * val_percent / 100))
        test_idx = np.uint(x.shape[0]) - train_idx - val_idx

        x_train = x[dataset_index[0:train_idx]]
        x_val = x[dataset_index[train_idx:train_idx + val_idx]]
        x_test = x[dataset_index[train_idx + val_idx:train_idx + val_idx + test_idx]]

        y_train = y[dataset_index[0:train_idx]]
        y_val = x[dataset_index[train_idx:train_idx + val_idx]]
        y_test = y[dataset_index[train_idx + val_idx:train_idx + val_idx + test_idx]]

        return x_train, x_val, x_test, y_train, y_val, y_test
