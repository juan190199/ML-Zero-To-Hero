import numpy as np

import progressbar
from utils.misc import bar_widgets


class NeuralNetwork():
    """
    Neural Network. Deep Learning base model
    """
    def __init__(self, optimizer, loss, validation_data=None):
        """

        :param optimizer:
        :param loss:
        :param validation_data:
        """
        self.optimizer = optimizer
        self.layers = []
        self.errors = {'training': [], 'validation': []}
        self.loss_function = loss()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {'X': X, 'y': y}

    def set_trainable(self, trainable):
        """
        Enables freezing of the weights of the network's layers
        :param trainable:
        :return:
        """
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """

        :param layer:
        :return:
        """
        # If not first hidden layer,
        # then set the input shape to the output shape of the last added layer.
        if self.layers:
            layer.set_input.shape(shape=self.layers[-1].output.shape())

        # If the layer has weights that need to be initialized
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        # Add layer to the network
        self.layers.append(layer)

    def test_on_batch(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        ...

    def train_on_batch(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        ...

    def fit(self, X, y, n_epochs, batch_size):
        """

        :param X:
        :param y:
        :param n_epochs:
        :param batch_size:
        :return:
        """
        ...

    def _forward_pass(self, X, training=True):
        """

        :param X:
        :param training:
        :return:
        """
        ...

    def _backward_pass(self, loss_grad):
        """

        :param loss_grad:
        :return:
        """
        ...

    def summary(self, name="Model Summary"):
        """

        :param name:
        :return:
        """
        ...

    def predict(self, X):
        """

        :param X:
        :return:
        """
        ...
