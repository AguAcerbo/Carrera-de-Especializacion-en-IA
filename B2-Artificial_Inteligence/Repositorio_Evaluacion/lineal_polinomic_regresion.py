import numpy as np
import matplotlib.pyplot as plt


class BaseModel(object):

    def __init__(self):
        self.model = None
        self.predicted = None

    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, x_train, y_train):
        w = y_train.mean()
        self.model = w

    def predict(self, x):
        self.predicted = np.ones(len(x)) * self.model

    def plot_model(self, x, y):
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.gca().set_title('Fitting curves')
        plt.plot(x, y, 'o')
        plt.plot(x, self.predicted, '-')


class LinearRegression(BaseModel):

    def fit(self, x_train, y_train):
        if len(x_train.shape) == 1:
            w = x_train.T.dot(y_train) / x_train.T.dot(x_train)
        else:
            # w = ((x_train.T*x_train)^-1)*x_train.T*y_train
            w = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        self.model = w

    def predict(self, x):
        self.predicted = np.sum(self.model.T * x, axis=1)

    def plot_model(self, x, y):
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.gca().set_title('Fitting curves')
        plt.plot(x, y, 'o')
        plt.plot(x, self.predicted, '-')


class LinearRegressionWithB(BaseModel):

    def fit(self, x_train, y_train):
        # x_expanded = [x_train, 1]
        x_expanded = np.vstack((x_train, np.ones(len(x_train)))).T
        # w = ((x_train.T*x_train)^-1)*x_train.T*y_train
        w = np.linalg.inv(x_expanded.T.dot(x_expanded)).dot(x_expanded.T).dot(y_train)
        self.model = w

    def predict(self, x):
        x_expanded = np.vstack((x, np.ones(len(x)))).T
        self.predicted = x_expanded.dot(self.model.T)

    def plot_model(self, x, y):
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.gca().set_title('Fitting curves')
        plt.plot(x, y, 'o')
        plt.plot(x, self.predicted, '-')
