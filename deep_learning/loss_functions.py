import sys
import numpy as np

sys.path.append('../')

from utils import accuracy_score, to_categorical


class Loss(object):
    def loss(self, y, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        return NotImplementedError()

    def acc(self, y, y_pred):
        return 0


##################################
###     Regression metrics     ###
##################################

class SquareLoss(Loss):
    def __init__(self):
        pass

    def __call__(self, y, y_pred, normalize=True):
        if normalize:
            return 0.5 * np.mean(np.power((y - y_pred), 2))
        else:
            return 0.5 * np.sum(np.power((y - y_pred), 2))

    def gradient(self, y, y_pred, normalize=True):
        if normalize:
            return -(y - y_pred) / y.shape[0]
        else:
            return -(y - y_pred)


######################################
###     Classification metrics     ###
######################################

class CrossEntropy(Loss):
    def __init__(self):
        pass

    def __call__(self, y, y_pred, normalize=False):
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        if normalize:
            return - np.mean(y * np.log(y_pred), axis=1)
        else:
            return -np.sum(y * np.log(y_pred), axis=1)

    def gradient(self, y, y_pred, normalize=False):
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        if normalize:
            return -(y - y_pred) / y.shape[0]
        else:
            return -(y - y_pred)

    def acc(self, y, y_pred):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
