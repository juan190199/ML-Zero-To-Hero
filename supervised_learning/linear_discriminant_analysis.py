import numpy as np

from utils.data_operation import calculate_covariance_matrix


class LDA():
    """

    """
    def __init__(self):
        self.w = None

    def transform(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.fit(X, y)
        # Project data onto vector
        X_transform = X.dot(self.w)
        return X_transform

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        # Separate data by class
        X1 = X[y == 0]
        X2 = X[y == 1]

        # Calculate the covariance of the two datasets
        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)
        cov_tot = cov1 + cov2

        # Calculate the mean of the two datasets
        mean1 = X1.mean(axis=0)
        mean2 = X2.mean(axis=0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Determine the vector which best separates the data by class when X is projected.
        # w = (mean1 - mean2) / (cov1 + cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred

