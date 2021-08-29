import math
import numpy as np

from utils.data_manipulation import (normalize, polynomial_features)


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


class L1_L2_Regularization():
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)


class Regression(object):
    """
    Base regression model. Models the relationship between a scalar dependent variable y and the independent
    variables X.

    """
    def __init__(self, n_iterations, learning_rate):
        """

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features, scale=1):
        """
        Initialize weights randomly

        :param n_features:

        :param scale: float
            scale of np.random.normal

        :return: self
        """
        limit = 1 / math.sqrt(n_features)
        # self.w = np.random.normal(loc=0, scale=scale,  size=(n_features, ))
        self.w = np.random.uniform(-limit, limit, size=(n_features, ))

    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target data

        :return: self
        """
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # Calculate L2 loss w.r.t. w
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Gradient of L2 loss w.r.t. w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        #Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """
    Linear model.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        """

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.

        :param gradient_descent: boolean
            True or false depending if gradient descent should be used when training.
            If false then we use batch optimization by least squares.
        """
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target values

        :return: self
        """
        # If not gradient descent => Normal equations
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)


class LassoRegression(Regression):
    """

    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        """

        :param degree: int
            The degree of the polynomial that the independent variable X will be transformed to.

        :param reg_factor: float
             The factor that will determine the amount of regularization and feature shrinkage.

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.
        """
        self.degree = degree
        self.regularization = L1_Regularization(alpha=reg_factor)
        super(LassoRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X, y)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(LassoRegression, self).predict(X)


class PolynomialRegression(Regression):
    """

    """
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        """

        :param degree: int
            The degree of the polynomial that the independent variable X will be transformed to.

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.
        """
        self.degree = degree
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)


class RidgeRegression(Regression):
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        """

        :param reg_factor: float
            The factor that will determine the amount of regularization and feature shrinkage.

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.
        """
        self.regularization = L2_Regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)


class PolynomialRidgeRegression(Regression):
    """

    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, gradient_descent=True):
        """

        :param degree:
        :param reg_factor:
        :param n_iterations:
        :param learning_rate:
        :param gradient_descent:
        """
        self.degree = degree
        self.regularization = L2_Regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)


class ElasticNet(Regression):
    """

    """
    def __init__(self, degree=1, reg_factor=0.05, n_iterations=3000, learning_rate=0.01):
        """

        :param degree:
        :param reg_factor:
        :param n_iterations:
        :param learning_rate:
        """
        self.degree = degree
        self.regularization = L1_L2_Regularization
