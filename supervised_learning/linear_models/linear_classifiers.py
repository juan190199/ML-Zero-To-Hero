import math
import numpy as np

from deep_learning import CrossEntropy
from supervised_learning.linear_models.regression import Regression, LinearRegression, RidgeRegression
from utils.data import make_diagonal, normalize, polynomial_features, to_categorical
from deep_learning.activation_functions import Sigmoid, Softmax


class SoftmaxRegression(Regression):
    """
    Softmax regression classifier
    """
    # ToDo: implement features class weight,
    #  solvers(lbfgs - multinomial, penalty(l2), newton-cg - multinomial, penalty(l2), newton-cholesky - penalty: l2,
    #  saga - multinomial, penalty(elasticnet, l1, l2))

    def __init__(self, max_iterations=1000, learning_rate=0.01, tol=1e-4, solver='gradient_descent'):
        self.allowed_solvers = ["gradient_descent"]
        self.degree = 1

        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        self.loss_function = CrossEntropy()

        super().__init__(
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            tol=tol,
            solver=solver
        )

    def initialize_weights(self, n_features, n_classes, scale=1):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, size=(n_features, n_classes))

    def fit(self, X, y, sample_weight=None):
        X = normalize(polynomial_features(X, degree=self.degree))
        X = np.insert(X, 0, 1, axis=1)
        sample_weight = make_diagonal(np.ones(X.shape[0])) if sample_weight is None else make_diagonal(sample_weight)

        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1], n_classes=len(np.unique(y)))

        if self.solver == 'gradient_descent':
            self.gradient_descent(
                np.sqrt(sample_weight).dot(X),
                np.sqrt(sample_weight).dot(to_categorical(y, len(np.unique(y))))
            )

    def predict_proba(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        X = np.insert(X, 0, 1, axis=1)

        logits = X.dot(self.w)
        probabilities = Softmax()(logits)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        for i in range(self.max_iterations):
            logits = X.dot(self.w)
            probabilities = Softmax()(logits)

            loss = self.loss_function.loss(y, probabilities)
            self.training_errors.append(loss)

            grad_w = np.dot(X.T, self.loss_function.gradient(y, probabilities))
            self.w -= self.learning_rate * grad_w


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
