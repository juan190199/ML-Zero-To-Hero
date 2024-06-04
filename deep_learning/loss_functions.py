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

    def loss(self, y, y_pred, normalize=True):
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

    def loss(self, y, y_pred, normalize=True):

        # Ensure y is one-hot encoded
        if len(y.shape) == 1 or y.shape[1] == 1:
            y = to_categorical(y)

        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        sample_losses = -np.sum(y * np.log(y_pred), axis=1)

        if normalize:
            return np.mean(sample_losses)
        else:
            return np.sum(sample_losses)

    def acc(self, y, y_pred):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))

    def gradient(self, y, y_pred, normalize=True):
        # Ensure y is one-hot encoded
        if len(y.shape) == 1 or y.shape[1] == 1:
            y = to_categorical(y)

        grad = y_pred - y
        if normalize:
            return grad / y.shape[0]
        else:
            return grad
