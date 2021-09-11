import math
import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Returns the mean squared error between y_true and y_pred

    :param y_true:
    :param y_pred:
    :return:
    """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_variance(X):
    """

    :param X:
    :return:
    """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance


def calculate_std_dev(X):
    """
    Calculate the standard deviations of the features in dataset X
    """
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def euclidean_distance(x1, x2):
    """
    :param x1: ndarray of shape (n_samples1, n_features)
    :param x2:ndarray of shape (n_samples2, n_features)
    :return:
    """
    distance = np.sqrt(np.sum(np.square(np.subtract(x1[:, None, :], x2)), axis=2))
    return distance


def accuracy_score(y, y_pred):
    """
    Compare ground truth y to predictions y_pred and return the accuracy

    :param y: ndarray of shape (n_samples, )
        Target values

    :param y_pred: ndarray of shape (n_samples, )
        Predicted values

    :return: float
        Accuracy of comparison between y and y_pred
    """
    accuracy = np.sum(y == y_pred, axis=0) / len(y)
    return accuracy


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


def calculate_correlation_matrix(X, Y=None):
    """
    Calculate the correlation matrix for the dataset X
    """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)


def logsumexp(arr, axis=0):
    """
    Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
