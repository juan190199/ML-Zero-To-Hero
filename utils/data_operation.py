import sys

import math
import numpy as np


def euclidean_distance(x1, x2):
    """
    :param x1: ndarray of shape (n_samples1, n_features)
    :param x2:ndarray of shape (n_samples2, n_features)
    :return:
    """
    distance = np.sqrt(np.sum(np.square(np.subtract(x1[:, None, :], x2)), axis=2))
    return distance


def calculate_covariance_matrix(X, Y=None):
    """

    :param X:
    :param Y:
    :return:
    """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)
