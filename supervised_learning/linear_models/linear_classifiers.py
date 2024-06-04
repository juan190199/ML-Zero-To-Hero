import math
import numpy as np

from deep_learning import CrossEntropy
from supervised_learning.linear_models.regression import LinearRegression, RidgeRegression
from utils.data import make_diagonal
from deep_learning.activation_functions import Sigmoid


class LogisticRegression(LinearRegression):
    """
    Logistic regression classifier
    """

    def __init__(self, max_iterations=1000, learning_rate=0.01, tol=1e-4, solver='gradient_descent'):
        super().__init__(max_iterations=max_iterations, learning_rate=learning_rate, solver=solver)
        self.tol = tol
        self.sigmoid = Sigmoid()
        self.cross_entropy = CrossEntropy()

    def fit(self, X, y, sample_weight=None):
        self.initialize_weights(X.shape[1])

        if self.solver == 'gradient_descent':
            self.gradient_descent(X, y)
        elif self.solver == 'newton_raphson':
            self.newton_raphson(X, y)

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w))

    def predict(self, X):
        return np.round(self.predict_proba(X)).astype(int)

    def gradient_descent(self, X, y):
        for i in range(self.max_iterations):
            y_pred = self.sigmoid(np.dot(X, self.w))
            loss = self.cross_entropy.loss(y, y_pred)
            grad = self.cross_entropy.gradient(y, y_pred)

            self.w -= self.learning_rate * np.dot(X.T, grad)

            if np.linalg.norm(grad) < self.tol:
                break

    def newton_raphson(self, X, y):
        for i in range(self.max_iterations):
            y_pred = self.sigmoid(np.dot(X, self.w))
            loss = self.cross_entropy.loss(y, y_pred)
            grad = self.cross_entropy.gradient(y, y_pred)
            diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.w)))
            hessian = np.dot(X.T, np.dot(diag_gradient, X))
            self.w -= np.linalg.inv(hessian).dot(np.dot(X.T, grad))

            if np.linalg.norm(grad) < self.tol:
                break


class RidgeClassifier(RidgeRegression):
    """
    Ridge regression classifier. Inherits from the RidgeRegression class
    """

    def __init__(self, reg_factor, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        super().__init__(reg_factor=reg_factor, n_iterations=n_iterations, learning_rate=learning_rate,
                         gradient_descent=gradient_descent)

    def fit(self, X, y):
        """
        Fit the RidgeRegression classifier to the training data.
        Args:
            X: ndarray of shape (n_samples, n_features)
                 Training data.

            y: ndarray of shape (n_samples, )
                 Target values.

        Returns: self

        """
        super(RidgeClassifier, self).fit(X, y)

    def predict(self, X):
        """
        Predict the class labels of the provided data.

        Args:
            X: ndarray of shape (n_samples, n_features)
                 Training data.

        Returns: ndarray of shape (n_samples, )
                 Predicted class labels.
        """
        y_pred = super(RidgeClassifier, self).predict(X)
        return np.where(y_pred >= 0, 1, -1)
