import numpy as np

import progressbar
from utils.misc import bar_widgets

from terminaltables import AsciiTable

from utils.data_manipulation import batch_iterator


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
            layer.set_input_shape(shape=self.layers[-1].output_shape())

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
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def train_on_batch(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function w.r.t. y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        """

        :param X:
        :param y:
        :param n_epochs:
        :param batch_size:
        :return:
        """
        for _ in self.progressbar(range(n_epochs)):

            batch_error = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)

        #     self.errors["training"].append(np.mean(batch_error))
        #
        #     if self.val_set is not None:
        #         val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
        #         self.errors["validation"].append(val_loss)
        #
        # return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, X, training=True):
        """

        :param X:
        :param training:
        :return:
        """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        # return layer_output

    def _backward_pass(self, loss_grad):
        """

        :param loss_grad:
        :return:
        """
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def summary(self, name="Model Summary"):
        """

        :param name:
        :return:
        """
        # Print model name
        print(AsciiTable([[name]]).table)
        # Network input shape (first layer's input shape)
        print("Input shape: %s" % str(self.layers[0].input_shape))
        # Iterate through network and get each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        # Print network configuration table
        print(AsciiTable(table_data).table)
        print("Total Parameters: %d\n" % tot_params)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        self._forward_pass(X, training=False)
