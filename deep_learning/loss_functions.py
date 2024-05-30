import sys
import numpy as np

sys.path.append('../')

from utils import accuracy_score


class Loss(object):
    def loss(self, y, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        return NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def acc(self, y, y_pred):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))

    def gradient(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y / y_pred) + (1 - y) / (1 - y_pred)
