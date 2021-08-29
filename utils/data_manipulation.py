import sys

import math
import numpy as np

from itertools import combinations_with_replacement


def polynomial_features(X, degree):
    """

    :param X: ndarray of shape (n_samples, n_features)
        Dataset

    :param degree: int


    :return:
    """
    n_samples, n_features = X.shape
    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs


def normalize(X, axis=-1, order=2):
    """
    Normalize dataset X

    :param X: ndarray of shape (n_samples, n_features)
        Dataset

    :param axis: int
        Specifies the axis of x along which to compute the vector norms

    :param order: int
        Order of the norm

    :return: ndarray of shape (n_samples, n_features)
        Normalized dataset
    """
    L2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    L2[L2 == 0] = 1
    return X / np.expand_dims(L2, axis)


def standardize(X):
    """
    Standardize dataset X

    :param X: ndarray of shape (n_samples, n_features)
        Dataset

    :return: ndarray of shape (n_samples, n_features)
        Standardized dataset
    """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std


def make_diagonal(x):
    """
    Converts a vector into a diagonal matrix
    :param x:
    :return:
    """
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m



