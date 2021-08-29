import math

import numpy as np


class L1_Regularization():
    """
    Regularization for Lasso Regression
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class L2_Regularization():
    """
    Regularization for Lasso Regression
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class Regression(object):
    # """
    # Base regression model. Models the relationship between a scalar dependent variable y and the independent
    # variables X.
    #
    # """
    # def __init__(self, n_iterations, learning_rate):
    #     self.n_iterations = n_iterations
    #     self.learning_rate = learning_rate
    #
    # def initialize_weights(self, n_features, scale=1):
    #     """
    #     Initialize weights randomly
    #
    #     :param n_features:
    #
    #     :param scale: float
    #         scale of np.random.normal
    #     :return:
    #     """
    #     limit = 1 / math.sqrt(n_features)
    #     self.w = np.random.normal(loc=0, scale=scale,  sizze=(n_features, ))
    #
    # def fit(self, X, y):
    #     """
    #
    #     :param X: ndarray of shape (n_samples, n_features)
    #         Training data
    #     :param y: ndarray of shape (n_samples, )
    #         Target data
    #     :return:
    #     """
    #     # Insert constant ones for bias weights
    #     X = np.insert(X, 0, 1, axis=1)
    #     self.training_errors = []
    #     self.initialize_weights(n_features=X.shape[1])
    #
    #     # Gradient descent for n_iterations
    #     for i in range(self.n_iterations):
    #         y_pred = X.dot(self.w)
    #         # Calculate L2 loss w.r.t. w
    #         mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.w))
    #         self.training_errors.append(mse)
    #         # Gradient of L2 loss w.r.t. w
    #         grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
    #         # Update the weights
    #         self.w -= self.learning_rate * grad_w
    #
    # def predict(self, X):
    #     """
    #
    #     :param X:
    #     :return:
    #     """
    #     #Insert constant ones for bias weights
    #     X = np.insert(X, 0, 1, axis=1)
    #     y_pred = X.dot(self.w)
    #     return y_pred


class LinearRegression(Regression):
    # """
    # Linear model
    # """
    # def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
    #     """
    #
    #     :param n_iterations:
    #     :param learning_rate:
    #     :param gradient_descent:
    #     """
    #     self.gradient_descent = gradient_descent
    #     # No regularization
    #     self.regularization = lambda x: 0
    #     self.regularization.grad = lambda x: 0
    #     super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)
    #
    # def fit(self, X, y):
    #     """
    #
    #     :param X:
    #     :param y:
    #     :return:
    #     """
    #     # If not gradient descent =>
    #     if not self.gradient_descent:
    #         pass
    #     else:
    #         super(LinearRegression, self).fit(X, y)


class LassoRegression(Regression):
    ...


class PolynomialRegression(Regression):
    ...


class RidgeRegression(Regression):
    ...


class PolynomialRidgeRegression(Regression):
    ...


class ElasticNet(Regression):
    ...
