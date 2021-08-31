import math
import numpy as np


def calculate_variance(X):
    """

    :param X:
    :return:
    """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance


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


def logsumexp(arr, axis=0):
    """
    Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
