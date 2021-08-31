import math
import numpy as np

from deep_learning.activation_functions import (Sigmoid, Softmax)
from deep_learning.loss_functions import CrossEntropy


class MultilayerPerceptron():
    """
    Multilayer Perceptron classifier. A fully-connected neural network with one hidden layer.
    """

    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        """

        :param n_hidden: int
            Number of processing nodes (neurons) in the hidden layer

        :param n_iterations: float
            The number of training iterations the algorithm will tune the weights for

        :param learning_rate: float
            Step length that will be used when updating the weights.
        """
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()

    def _initialize_weights(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target data

        :return:
        """

    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target data

        :return:
        """
        n_samples, n_features = np.shape(X)
        _, n_outputs = y.shape

        # Hidden layer
        limit = 1 / math.sqrt(n_features)
        self.w1 = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.b1 = np.zeros((1, self.n_hidden))

        # Output layer
        limit = 1 / math.sqrt(self.n_hidden)
        self.w2 = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.b2 = np.zeros((1, n_outputs))

    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target data

        :return:
        """
        # Forward propagation
        z1 = X.dot(self.w1) + self.b1
        a1 = self.hidden_activation(z1)
        z2 = a1.dot(self.w2) + self.b2
        y_pred = self.output_activation(z2)

        # Backpropagation
        ...

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return:
        """
        # Forward propagation
        z1 = X.dot(self.w1) + self.b1
        a1 = self.hidden_activation(z1)
        z2 = a1.dot(self.w2) + self.b2
        y_pred = self.output_activation(z2)
        return y_pred



