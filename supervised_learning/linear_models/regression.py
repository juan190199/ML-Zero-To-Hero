import math
import numpy as np

from utils.data_manipulation import (normalize, polynomial_features, make_diagonal)
from utils.data_operation import rescale_data
from utils.data_operation import calculate_covariance_matrix


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
    allowed_solvers = []

    def __init__(self, max_iterations=1000, learning_rate=0.01, tol=1e-4, solver='gradient_descent'):
        """

        :param max_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.

        :param tol: float

        :param solver: str

        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tol = tol

        if solver not in self.allowed_solvers:
            raise ValueError('`Solver not supported.')
        self.solver = solver

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
        self.w = np.random.uniform(-limit, limit, size=(n_features,))

    def fit(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        # Preprocess data
        X = normalize(polynomial_features(X, degree=self.degree))
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        sample_weight = make_diagonal(np.ones(X.shape[0])) if sample_weight is None else make_diagonal(sample_weight)

        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        if self.solver == 'normal_equations':
            self.normal_equations(X, y, sample_weight=sample_weight)
        elif self.solver == 'gradient_descent':
            self.gradient_descent(np.sqrt(sample_weight).dot(X), np.sqrt(sample_weight).dot(y))
        elif self.solver == 'coordinate descent':
            self.coordinate_descent(np.sqrt(sample_weight).dot(X), np.sqrt(sample_weight).dot(y))
        elif self.solver == 'lar':
            self.lar(X, y)
        elif self.solver == 'omp':
            self.omp(X, y)

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        # Preprocess data
        X = normalize(polynomial_features(X, degree=self.degree))
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

    def gradient_descent(self, X, y):
        """
        Perform gradient descent optimization.

        Args:
            X: ndarray of shape (n_samples, n_features)
                Training data.

            y: ndarray of shape (n_samples, )
                Target data

        Returns: self

        """
        for i in range(self.max_iterations):
            y_pred = X.dot(self.w)
            # Calculate L2 loss w.r.t. w
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.r54egularization(self.w))
            self.training_errors.append(mse)
            # Gradient of L2 loss w.r.t. w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

            # Check convergence criterion
            if np.max(np.abs(grad_w)) < self.tol:
                break

    def normal_equations(self, X, y, sample_weight):
        U, S, V = np.linalg.svd(X.T.dot(sample_weight).dot(X) + self.regularization(self.w) * np.identity(X.shape[1]))
        S = np.diag(S)
        X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
        self.w = X_sq_reg_inv.dot(X.T).dot(sample_weight).dot(y)

    def coordinate_descent(self, X, y):
        """
        Coordinate descent optimization algorithm.

        Args:
            X: ndarray of shape (n_samples, n_features)
                Training data.

            y: ndarray of shape (n_samples, )
                Target data

        Returns: self

        """
        for i in range(self.max_iterations):
            # Store old weights for convergence check:
            w_old = self.w.copy()

            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.w))
            self.training_errors.append(mse)

            for j in range(X.shape[1]):
                w_j_old = self.w[j]
                w_except_j = w_old[np.arange(w_old.shape[0]) != j]
                X_except_j = X[:, np.arange(X.shape[1]) != j]

                # Calculate the partial derivative of the objective function w.r.t. the j-th coordinate of the weight vector.
                rho_j = X[:, j].dot(y - w_except_j.dot(X_except_j))

                # Update the j-th coordinate of the weight vector
                self.w[j] = self.soft_thresholding_operator(rho_j, self.regularization.alpha / 2)

                if self.w[j] != w_j_old:
                    y_pred += X[:, j] * (self.w[j] - w_j_old)

            if np.linalg.norm(self.w - w_old) < self.tol:
                break

    def soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0.0


class LinearRegression(Regression):
    """

    """

    def __init__(self, max_iterations=100, learning_rate=0.001, solver='gradient_descent'):
        """

        Args:
            max_iterations:
            learning_rate:
            solver:
        """
        self.allowed_solvers = ["normal_equations", "gradient_descent", "coordinate descent"]
        self.solver = solver

        self.degree = 1

        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        super(LinearRegression, self).__init__(max_iterations=max_iterations, learning_rate=learning_rate,
                                               solver=solver)

    def fit(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:
                If None, then samples are equally weighted. Otherwise, sample_weight is used to weight the observations.
                Common choice of sample weights is exp(-(x^{(i)} - x)^2 / 2 * tau^2), where x is the input for which
                the prediction is to be made.

        Returns:

        """
        super(LinearRegression, self).fit(X, y, sample_weight)


class RidgeRegression(Regression):
    """
    Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.
    """

    def __init__(self, reg_factor, max_iterations=1000, learning_rate=0.001, solver="gradient_descent"):
        """

        Args:
            reg_factor:
            max_iterations:
            learning_rate:
            solver:
        """
        self.allowed_solvers = ["normal_equations", "gradient_descent", "coordinate_descent"]
        self.solver = solver

        self.degree = 1

        self.reg_factor = reg_factor
        self.regularization = L2_Regularization(alpha=self.reg_factor)

        super(RidgeRegression, self).__init__(max_iterations=max_iterations, learning_rate=learning_rate)

    def fit(self, X, y, sample_weights=None):
        """

        Args:
            X:
            y:
            sample_weights:

        Returns:

        """
        super(RidgeRegression, self).fit(X, y, sample_weights)


class LassoRegression(Regression):
    """
    Linear regression model with a regularization factor which does both variable selection
    and regularization. Model that tries to balance the fit of the model with respect to the training
    data and the complexity of the model. A large regularization factor with decreases the variance of the model.
    """

    def __init__(self, reg_factor, degree=1, max_iterations=3000, learning_rate=0.01, solver="coordinate_descent"):
        """

        :param degree: int
            The degree of the polynomial that the independent variable X will be transformed to.

        :param reg_factor: float
             The factor that will determine the amount of regularization and feature shrinkage.

        :param max_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.
        """
        self.allowed_solvers = ["coordinate_descent", "lar", "omp"]
        self.solver = solver

        self.degree = degree

        self.degree = degree
        self.regularization = L1_Regularization(alpha=reg_factor)

        super(LassoRegression, self).__init__(max_iterations=max_iterations, learning_rate=learning_rate, solver=solver)

    def fit(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        X = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(LassoRegression, self).predict(X)


class ElasticNet(Regression):
    """
    Regression where a combination of l1 and l2 regularization are used. The
    ratio of their contributions are set with the 'l1_ratio' parameter.
    """

    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, max_iterations=3000, learning_rate=0.01):
        """

        Args:
            degree:
            reg_factor:
            l1_ratio:
            max_iterations:
            learning_rate:
        """
        self.allowed_solvers = ["coordinate_descent"]
        self.solver = "coordinate_descent"

        self.degree = degree
        self.regularization = L1_L2_Regularization(alpha=reg_factor, l1_ratio=l1_ratio)

        super(ElasticNet, self).__init__(max_iterations=max_iterations, learning_rate=learning_rate, solver=solver)

    def fit(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        super(ElasticNet, self).fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """

        Args:
            X:

        Returns:

        """
        return super(ElasticNet, self).predict(X)


class PolynomialRegression(Regression):
    """

    """

    def __init__(self, degree, max_iterations=3000, learning_rate=0.001, solver='gradient_descent'):
        """

        Args:
            degree:
            max_iterations:
            learning_rate:
        """
        self.allowed_solvers = ["normal_equations", "gradient_descent", "coordinate descent"]
        self.solver = solver

        self.degree = degree

        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        super(PolynomialRegression, self).__init__(max_iterations=max_iterations, learning_rate=learning_rate)

    def fit(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        super(PolynomialRegression, self).fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """

        Args:
            X:

        Returns:

        """
        return super(PolynomialRegression, self).predict(X)


class PolynomialRidgeRegression(Regression):
    """

    """

    def __init__(self, degree, reg_factor, max_iterations=3000, learning_rate=0.01, solver='gradient_descent'):
        """

        Args:
            degree:
            reg_factor:
            max_iterations:
            learning_rate:
            solver:
        """
        self.allowed_solvers = ["normal_equations", "gradient_descent", "coordinate descent"]
        self.solver = solver

        self.degree = degree

        self.regularization = L2_Regularization(alpha=reg_factor)

        super(PolynomialRidgeRegression, self).__init__(max_iterations=max_iterations, learning_rate=learning_rate)

    def fit(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        super(PolynomialRidgeRegression, self).fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        return super(PolynomialRidgeRegression, self).predict(X)
