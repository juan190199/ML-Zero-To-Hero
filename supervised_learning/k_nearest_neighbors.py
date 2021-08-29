import numpy as np

from utils.data_operation import euclidean_distance


class KNN():
    """
    K Nearest Neighbors classifier
    """

    def __init__(self, k=5, C=10):
        """

        :param k: int
            The number of closest neighbors that will determine the class of the sample that we wish to predict.
        :param C: int, default=10
            Number of classes present in target values
        """
        self.k = k
        self.C = C

    def _get_knn(self, X_train, X_test):
        """

        :param X_test: ndarray of shape (m, n_features)
            Test data

        :param X_train: ndarray of shape (n, n_features)
            Training data

        :return: ndarray of shape (k, n_samples1)
            Indices of the k nearest neighbors for each test sample
        """
        # Calculate euclidean distances between training and test data
        dist = euclidean_distance(X_train, X_test)
        # Find k nearest neighbors for each test data point
        # ndarray (k, n_samples2) containing in each column the idx of knns' in training set
        idx = np.argsort(dist, axis=0)[:self.k, :]
        return idx

    def predict(self, X_test, X_train, y_train):
        """

        :param X_test: ndarray of shape (n_samples2, n_features)
            Test data

        :param X_train: ndarray of shape (n_samples1, n_features)
            Training data

        :param y_train: ndarray of shape (n_samples1, )
            Target values for training data

        :return: ndarray of shape (n_samples2, )
            Predicted labels for test data
        """
        idx = self._get_knn(X_train, X_test)
        # ndarray (k, n_samples2) containing in each column the idx of knns' in training set
        neighbors = np.take(y_train, idx)
        # ndarray (C, n_samples2) containing votes for each class per test point
        election = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.C), axis=0, arr=neighbors)
        # ndarray (n_samples2, ) containing label prediction for test set
        y_pred = np.argmax(election, axis=0)
        return y_pred

