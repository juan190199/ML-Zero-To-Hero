import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot

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


def batch_iterator(X, y=None, batch_size=64, drop_last=False):
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
            if drop_last and end - begin < batch_size:
                continue
            yield X[begin:end], y[begin:end]
        else:
            if drop_last and end - begin < batch_size:
                continue
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


def rescale_data(X, y, sample_weight):
    """
    Rescale data sample-wise by square root of sample weight
    :param X:
    :param y:
    :param sample_weight:
    :return:
    """
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)

    return X, y
