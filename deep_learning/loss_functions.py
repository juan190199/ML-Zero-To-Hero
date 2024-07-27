import sys
import numpy as np

sys.path.append('../')

from utils.metrics import accuracy_score
from utils.data import to_categorical


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
    def __init__(self, class_weights=None):
        """

        Args:
            class_weights: dict or np.array
        """
        if isinstance(class_weights, dict):
            self.class_weights = class_weights
        elif isinstance(class_weights, (np.ndarray, list)):
            self.class_weights = np.array(class_weights)
        elif class_weights is None:
            self.class_weights = None
        else:
            raise ValueError("class_weights must be a dictionary, an array, or None.")

    def __call__(self, y, y_pred, normalize=False):
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if self.class_weights is not None:
            if isinstance(self.class_weights, dict):
                # Convert class weights dictionary to an array
                class_weights = np.array([self.class_weights[i] for i in np.argmax(y, axis=1)])
            else:
                class_weights = self.class_weights[np.argmax(y, axis=1)]

            loss = -np.sum(class_weights * y * np.log(y_pred), axis=1)
        else:
            loss = -np.sum(y * np.log(y_pred), axis=1)

        if normalize:
            return np.mean(loss)
        else:
            return np.sum(loss)

    def gradient(self, y, y_pred, normalize=False):
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if self.class_weights is not None:
            if isinstance(self.class_weights, dict):
                # Convert class weights dictionary to an array
                class_weights = np.array([self.class_weights[i] for i in np.argmax(y, axis=1)])
            else:
                class_weights = self.class_weights[np.argmax(y, axis=1)]

            # Calculate weighted gradient
            gradient = -(y - y_pred) * class_weights[:, np.newaxis]
        else:
            gradient = -(y - y_pred)

        if normalize:
            gradient /= y.shape[0]

        return gradient

    def acc(self, y, y_pred):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
