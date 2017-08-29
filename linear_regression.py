import numpy as np
from model import Model
from utils import *

DEFAULT_REG = 2


class LinearRegressionModel(Model):
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, reg=DEFAULT_REG):
        y_one_hotted = one_hot(y_train)
        _, d = X_train.shape
        lambda_I = np.zeros((d, d))
        np.fill_diagonal(lambda_I, reg)

        x_transpose = np.transpose(X_train)
        x_transpose_x = np.dot(x_transpose, X_train)
        inversed = np.linalg.inv(np.add(lambda_I, x_transpose_x))

        x_transpose_y = np.dot(x_transpose, y_one_hotted)
        self.model = np.dot(inversed, x_transpose_y)

    def predict(self, data):
        result = np.dot(data, self.model)
        length = len(result)
        y = np.zeros(length)
        for i in range(length):
            y[i] = np.argmax(result[i])
        return y
