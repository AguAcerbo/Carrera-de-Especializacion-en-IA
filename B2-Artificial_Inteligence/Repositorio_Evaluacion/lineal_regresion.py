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

# # y = sin(x)
# amt_points = 36
# x = np.linspace(0, 360, num=amt_points)
# y = np.sin(x * np.pi / 180.)
# noise = np.random.normal(0, .1, y.shape)
# noisy_y = y + noise
#
# x_train = x
# y_train = noisy_y
#
# x_10 = dataset_manipulation(x, polinomy_grade=1, bias=True)
# regression = LinearRegression()
#
# regression.fit(x_10,y_train)
# W_10 = regression.model
# regression.predict(x_10)
#
# y_predicted = regression.predicted
# regression.plot_model(x_train,y_train)