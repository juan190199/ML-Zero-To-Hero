import math
import numpy as np

import progressbar

from deep_learning.loss_functions import SquareLoss
from deep_learning.activation_functions import Sigmoid

from utils.misc import bar_widgets


class Perceptron():
    """
    The Perceptron. One layer neural network classifier
    """
    def __init__(self, n_iterations=20000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        """

        :param n_iterations: float
            Number of training iterations the algorithm will tune the weights for

        :param activation_function: class
            The activation that shall be used for each neuron
            Possible choices: Sigmoid, ExpLU, ReLU, LeakyReLU, SoftPlus, TanH

        :param loss: class
            The loss function used to assess the model's performance.
            Possible choices: SquareLoss, CrossEntropy

        :param learning_rate: float
            Step length that will be used when updating the weights
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.activation_function = activation_function()
        self.loss = loss()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target data

        :return: self
        """
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.b = np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iterations)):
            # Calculate outputs
            linear_output = X.T.dot(self.w) + self.b
            y_pred = self.activation_function(linear_output)
            # Calculate the loss gradient w.r.t. the input of the activation function
            error_gradient = self.loss.gradient(y, y_pred) * self.activation_function.gradient(linear_output)
            # Calculate the gradient of the loss w.r.t. each weight
            grad_wrt_w = X.T.dot(error_gradient)
            grad_wrt_b = np.sum(error_gradient, axis=0, keepdims=True)
            # Update weights
            self.w -= self.learning_rate * grad_wrt_w
            self.b -= self.learning_rate * grad_wrt_b

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predicted values
        """
        y_pred = self.activation_function(X.dot(self.w) + self.b)
        return y_pred


