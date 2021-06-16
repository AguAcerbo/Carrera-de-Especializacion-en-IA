import numpy as np


class LogisticRegression(object):

    def __init__(self):
        self.model = None
        self.predicted = None
        self.losses = None

    # definimos la función sigmoid para entrenamiento y las predicciones
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # definimos la función loss para reportarla cada cierta cantidad de epochs
    def loss(self, y, y_hat):
        loss = np.mean(-y * (np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))
        return loss

    def fit(self, x, y, lr, b, epochs, bias=True, log=100, verbose=True):

        # si decidimos utilizar bias, agregamos como siempre una columna con '1' al dataset de entrada
        if bias:
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        # inicializamos aleatoriamente los pesos
        m = x.shape[1]
        w = np.random.randn(m).reshape(m, 1)

        loss_list = []

        # corremos Mini-Batch para optimizar los parámetros
        for j in range(epochs):
            idx = np.random.permutation(x.shape[0])
            x_train = x[idx]
            y_train = y[idx]
            batch_size = int(len(x_train) / b)

            for i in range(0, len(x_train), batch_size):
                end = i + batch_size if i + batch_size <= len(x_train) else len(x_train)
                batch_x = x_train[i: end]
                batch_y = y_train[i: end]

                prediction = self.sigmoid(np.sum(np.transpose(w) * batch_x, axis=1))
                error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)
                grad_sum = np.sum(error * batch_x, axis=0)
                grad_mul = 1 / batch_size * grad_sum
                gradient = np.transpose(grad_mul).reshape(-1, 1)

                w = w - (lr * gradient)

            l_epoch = self.loss(y_train, self.sigmoid(np.dot(x_train, w)))
            loss_list.append(l_epoch)
            if verbose:
                if j % log == 0:
                    print("Epoch: {}, Loss: {}".format(j, l_epoch))

        self.model = w
        self.losses = loss_list

    def predict(self, x):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        p = self.sigmoid(x @ self.model)
        mask_true = p >= 0.5
        mask_false = p < 0.5
        p[mask_true] = 1
        p[mask_false] = 0
        return p
