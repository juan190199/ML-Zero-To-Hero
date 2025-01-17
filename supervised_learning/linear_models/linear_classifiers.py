import math
import numpy as np

from deep_learning.loss_functions import CrossEntropy
from supervised_learning.linear_models.regression import Regression, LinearRegression, RidgeRegression, \
    L2_Regularization
from utils.data import make_diagonal, normalize, polynomial_features, to_categorical
from deep_learning.activation_functions import Sigmoid, Softmax


class SoftmaxRegression(Regression):
    """
    Softmax regression classifier
    """

    #  solvers: lbfgs - multinomial, penalty(l2), newton-cg - multinomial, penalty(l2), newton-cholesky - penalty: l2,
    #  saga - multinomial, penalty(elasticnet, l1, l2))

    def __init__(
            self,
            max_iterations=1000,
            learning_rate=0.01,
            tol=1e-4,
            solver='gradient_descent',
            class_weights=None

    ):
        self.allowed_solvers = ["gradient_descent"]
        self.degree = 1

        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        self.class_weights = class_weights

        self.loss_function = CrossEntropy(class_weights=self.class_weights)

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


class RidgeClassifier(Regression):
    def __init__(
            self,
            reg_factor,
            max_iterations=1000,
            learning_rate=0.01,
            tol=1e-4,
            solver='gradient_descent',
            class_weights=None
    ):
        self.solver = solver
        self.degree = 1
        self.reg_factor = reg_factor
        self.regularization = L2_Regularization(alpha=self.reg_factor)
        self.class_weights = class_weights
        self.loss_function = CrossEntropy(class_weights=self.class_weights)

        super(RidgeClassifier, self).__init__(
            max_iterations=max_iterations,
            learning_rate=learning_rate, tol=tol,
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

            loss = self.loss_function.loss(y, probabilities) + self.regularization(self.w)
            self.training_errors.append(loss)

            grad_w = np.dot(X.T, self.loss_function.gradient(y, probabilities)) + self.regularization(self.w)
            self.w -= self.learning_rate * grad_w
