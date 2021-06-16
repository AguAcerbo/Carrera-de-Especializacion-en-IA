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

    def fit(self, x, y):
        w = y.mean()
        self.model = w

    def predict(self, x):
        self.predicted = np.ones(len(x)) * self.model
        return np.ones(len(x)) * self.model

    def plot_model(self, x, y):
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.gca().set_title('Fitting curves')
        plt.plot(x, y, 'o')
        plt.plot(x, self.predicted, '-')


class LinearRegression(BaseModel):

    def fit(self, x, y):
        if len(x.shape) == 1:
            w = x.T.dot(y) / x.T.dot(x)
        else:
            w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        self.model = w

    def predict(self, x):
        self.predicted = np.sum(self.model.T * x, axis=1)
        return np.sum(self.model.T * x, axis=1)

    def plot_model(self, x, y):
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.gca().set_title('Fitting curves')
        plt.plot(x, y, 'o')
        plt.plot(x, self.predicted, '-')


class LinearRegressionWithB(BaseModel):

    def fit(self, x, y):
        x_expanded = np.vstack((x, np.ones(len(x)))).T
        w = np.linalg.inv(x_expanded.T.dot(x_expanded)).dot(x_expanded.T).dot(y)
        self.model = w

    def predict(self, x):
        x_expanded = np.vstack((x, np.ones(len(x)))).T
        self.predicted = x_expanded.dot(self.model.T)
        return x_expanded.dot(self.model.T)

    def plot_model(self, x, y):
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.gca().set_title('Fitting curves')
        plt.plot(x, y, 'o')
        plt.plot(x, self.predicted, '-')
