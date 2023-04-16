import sys

import math
import numpy as np

from itertools import combinations_with_replacement


class Node:
    pass


class Tree:
    def __init__(self):
        self.root = Node()

    def find_leaf(self, x):
        node = self.root
        while hasattr(node, 'feature'):
            j = node.feature
            if x[j] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node


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

    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new


def shuffle_data(X, y, seed=None):
    """

    :param X:
    :param y:
    :param seed:
    :return:
    """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def batch_iterator(X, y=None, batch_size=64):
    """

    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


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
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


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


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """
    Split data into train and test sets
    :param X:
    :param y:
    :param test_size:
    :param shuffle:
    :param seed:
    :return:
    """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data with the ratio specified by test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def to_categorical(x, n_col=None):
    """
    One-hot encoding of nominal values

    :param x:
    :param n_col:
    :return:
    """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def to_nominal(x):
    """
    Conversion from one-hot encoding to nominal

    :param x:
    :return:
    """
    return np.argmax(x, axis=1)


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


# def KFold(X, y, k, shuffle=True):
#     """
#
#     :param X:
#     :param y:
#     :param k:
#     :param shuffle:
#     :return:
#     """
#     if shuffle:
#         X, y = shuffle_data(X, y)
#
#     n_samples = len(y)
#     left_overs = {}
#     n_left_overs = (n_samples % k)
#     if n_left_overs != 0:
#         left_overs['X'] = X[-n_left_overs:]
#         left_overs['y'] = y[-n_left_overs:]
#         X = X[:-n_left_overs]
#         y = y[:-n_left_overs]
#
#     X_split = np.split(X, k)
#     y_split = np.split(y, k)
#
#     sets = []
#     for i in range(k):
#         X_test, y_test = X_split[i], y_split[i]
#         X_train = np.concatenate(X_split[:i] + X_split[i + 1:], axis=0)
#         y_train = np.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
#         sets.append([X_train, X_test, y_train, y_test])
#
#     # Add left over samples to last set as training samples
#     if n_left_overs != 0:
#         np.append(sets[-1][0], left_overs['X'], axis=0)
#         np.append(sets[-1][2], left_overs['y'], axis=0)
#
#     return np.array(sets)


class KFoldCV:
    """
    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds;
    each fold is then used once as a validation while the k-1 remaining folds form the trraining set
    """

    def __init__(self, n_folds, shuffle=True, seed=12345):
        self.seed = seed
        self.shuffle = shuffle
        self.n_folds = n_folds

    def split(self, data_set):
        """
        Create mask for test set.
        :param data_set: data set
        :return:
        """
        # Shuffle modifies indices inplace
        n_samples = data_set.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            # A fixed seed and a fixed series of calls to ‘RandomState’ methods using the same parameters
            # will always produce the same results
            rstate = np.random.RandomState(self.seed)
            rstate.shuffle(indices)

        for test_mask in self._iter_test_masks(n_samples, indices):
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield train_index, test_index

    def _iter_test_masks(self, n_samples, indices):
        """
        Create test mask
        :param n_samples:
        :param indices:
        :return:
        """
        # If n_samples cannot be evenly split,
        # rest of samples have to be distributed between folds beginning with the leading one
        fold_sizes = (n_samples // self.n_folds) * np.ones(self.n_folds, dtype=np.int_)
        fold_sizes[:n_samples % self.n_folds] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test_mask = np.zeros(n_samples, dtype=np.bool_)
            test_mask[test_indices] = True
            yield test_mask
            current = stop