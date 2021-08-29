import numpy as np

from utils.data_operation import euclidean_distance


class KNN():
    """

    """

    def __init__(self, k=5, C=10):
        """

        :param k:
        """
        self.k = k

    def _get_knn(self, X_train, X_test):
        """

        :param X_train:
        :param X_test:
        :return:
        """
        # Find k nearest neighbors for each test data point
        dist = euclidean_distance(X_train, X_test)
        # ndarray (k, m) containing in each column the idx of knns' in training set
        idx = np.argsort(dist, axis=0)[:self.k, :]
        return idx

    def predict(self, X_test, X_train, y_train):
        """

        :param X_test: ndarray of shape (m, n_features)
            Test_matrix

        :param X_train: ndarray of shape (n, n_features)

        :param y_train:
        :return:
        """
        idx = self._get_knn(X_train, X_test)
        # ndarray (k, m) containing in each column the idx of knns' in training set
        neighbors = np.take(y_train, idx)
        # ndarray (C, m) containing votes for each class per test point
        election = np.apply_along_axis(lambda x: np.bincount(x, minlength=C), axis=0, arr=neighbors)
        # ndarray (m, ) containing label prediction for test set
        y_pred = np.argmax(election, axis=0)
        return y_pred

