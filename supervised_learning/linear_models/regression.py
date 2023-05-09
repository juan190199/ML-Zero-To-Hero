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
        self.w = np.random.uniform(-limit, limit, size=(n_features,))

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
        # Insert constant ones for bias weights
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
            True or false depending on if gradient descent should be used when training.
            If false then we use batch optimization by least squares.

        :return: self
        """
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y, sample_weights=None):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target values

        :param sample_weights: array_like of shape (n_samples, )
            If None, then samples are equally weighted. Otherwise, sample_weight is used
            to weight the observations.
            Common choice of sample weights is exp(-(x^{(i)} - x)^2 / 2 * tau^2), where x is the input for which
            the prediction is to be made.

        :return: self
        """
        sample_weights = make_diagonal(np.ones(X.shape[0])) if sample_weights is None else make_diagonal(sample_weights)
        # If not gradient descent => Normal equations
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudo-inverse)
            U, S, V = np.linalg.svd(X.T.dot(sample_weights).dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            # Calculate weights by normal equation
            self.w = X_sq_reg_inv.dot(X.T).dot(sample_weights).dot(y)

        else:
            super(LinearRegression, self).fit(np.sqrt(sample_weights).dot(X), np.sqrt(sample_weights).dot(y))


class RidgeRegression(Regression):
    """
    Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.
    """

    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001, gradient_descent=True):
        """

        :param reg_factor: float
            The factor that will determine the amount of regularization and feature shrinkage.

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.
        """
        self.gradient_descent = gradient_descent
        self.reg_factor = reg_factor
        self.regularization = L2_Regularization(alpha=self.reg_factor)
        super(RidgeRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

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
            U, S, V = np.linalg.svd(X.T.dot(X) + self.regularization * np.identity(X.shape[0]))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(RidgeRegression, self).fit(X, y)


class LassoRegression(Regression):
    """
    Linear regression model with a regularization factor which does both variable selection
    and regularization. Model that tries to balance the fit of the model with respect to the training
    data and the complexity of the model. A large regularization factor with decreases the variance of the model.
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

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target values

        :return: self
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X, y)

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

    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, learning_rate=0.01):
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
        self.regularization = L1_L2_Regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target values

        :return: self
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(ElasticNet, self).predict(X)


class LARS(Regression):
    def __init__(self, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, learning_rate=0.01, min_error_dif=1e-6):
        self.regularization = L1_L2_Regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        self.active_set = []
        self.coefficients = None
        self.min_error_dif = min_error_dif
        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        n_features = X.shape[1]
        self.initialize_weights(n_features=n_features)

        # Initialize the active set
        self.active_set = []
        self.coefficients = np.zeros(n_features)

        # Loop over the number of iterations
        for i in range(self.n_iterations):
            # Calculate the correlations between the features and the residuals
            correlations = X.T.dot(y - X.dot(self.coefficients))

            # Find the features with the maximum absolute correlation
            j = np.argmax(np.abs(correlations))
            sign = np.sign(correlations[j])

            # If the feature is not in the active set, add it to the active set
            if j not in self.active_set:
                self.active_set.append(j)

            # Calculate the current angle between the residual and the feature
            X_active = X[:, self.active_set]

            # projection = X_active.dot(np.linalg.inv(X_active.T.dot(X_active))).dot(X_active.T).dot(y)
            # Calculate the Cholesky decomposition of X_active.T.dot(X_active)
            L = np.linalg.cholesky(X_active.T.dot(X_active))
            # Solve for the inverse of X_active.T.dot(X_active) using the Cholesky decomposition
            inv_XTX = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(X_active.shape[1])))
            # Compute the projection matrix using the inverse of X_active.T.dot(X_active)
            projection = X_active.dot(inv_XTX).dot(X_active.T).dot(y)

            residual = y - projection
            angles = X.T.dot(residual)
            current_angle = np.abs(angles[j])

            # Calculate the step size and update the coefficients
            step_size = self.learning_rate / n_features * sign
            self.coefficients[self.active_set] += step_size

            # Update the training error
            y_pred = X.dot(self.coefficients)
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.coefficients))
            self.training_errors.append(mse)

            # Check for minimum improvement in training error
            if len(self.training_errors) > 0:
                error_dif = self.training_errors[-2] - mse
                if error_dif < self.min_error_dif:
                    break

            # If the angle between the residual and the feature is small, remove the feature from the active set
            if current_angle < 1e-15:
                self.active_set.remove(j)
                self.coefficients[j] = 0
                if len(self.active_set) == 0:
                    break


class PolynomialRegression(Regression):
    """
    Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.
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

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target values

        :return: self
        """
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)


class PolynomialRidgeRegression(Regression):
    """
    Similar to regular ridge regression except that the data is transformed to allow
    for polynomial regression.
    """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, gradient_descent=True):
        """

        :param degree: int
            The degree of the polynomial that the independent variable X will be transformed to.

        :param reg_factor: float
             The factor that will determine the amount of regularization and feature shrinkage.

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for.

        :param learning_rate: float
            The step length that will be used when updating the weights.

        :param gradient_descent: boolean
            True or false depending on if gradient descent should be used when training.
            If false then we use batch optimization by least squares.
        """
        self.degree = degree
        self.regularization = L2_Regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target values

        :return: self
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)
