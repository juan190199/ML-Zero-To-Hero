import math
import numpy as np

from utils.data_manipulation import make_diagonal
from deep_learning.activation_functions import Sigmoid


class LogisticRegression():
    """

    """

    def __init__(self, learning_rate=.1, gradient_descent=True):
        """

        :param learning_rate: float
            The step length that will be taken when following the negative gradient during training.

        :param gradient_descent: boolean
            True or false depending if gradient descent should be used when training.
            If false then we use batch optimization by least squares.
        """
        self.w = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def initialize_parameters(self, X):
        n_features = X.shape[1]
        # Initialize parameters between [-1/sqrt(d), 1/sqrt(d)]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, size=(n_features,))

    def fit(self, X, y, n_iterations=4000):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target data

        :return: self
        """
        self.initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = self.sigmoid(X.dot(self.w))
            if self.gradient_descent:
                self.w -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.w)))
                # Batch optimization
                self.w = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T)\
                    .dot(diag_gradient.dot(X).dot(self.w) + y - y_pred)

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        y_predict = np.round(self.sigmoid(X.dot(self.w))).astype(int)
        return y_predict
