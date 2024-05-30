import numpy as np

import pickle
from tqdm import tqdm

from terminaltables import AsciiTable

from utils import batch_iterator


class NeuralNetwork():
    """
    Neural Network. Deep Learning base model
    """

    def __init__(self, optimizer, loss, metrics=[]):
        """
        Initialize the neural network
        Args:
            optimizer: optimizer object - optimizer to be used for trainings
            loss: loss object - loss function to be used for training
            metrics: list of strings - metrics to be used for training
        """
        self.optimizer = optimizer
        self.layers = []
        self.errors = {'training': [], 'validation': []}
        self.loss_function = loss()
        self.metrics = metrics
        # self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def set_trainable(self, trainable):
        """
        Enable/disable trainable layers
        Args:
            trainable: True - enable trainable layers

        Returns:

        """
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """
        Add a layer to the neural network.
        Args:
            layer: layer object - layer to be added

        Returns:

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
        Test the neural network on a single batch of data
        Args:
            X: ndarray - input data
            y: ndarray - target data

        Returns:
            loss: float - test loss
            acc: float - test accuracy

        """
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        metrics = {metric.name: metric(y, y_pred) for metric in self.metrics}

        return loss, metrics

    def train_on_batch(self, X, y):
        """
        Train the neural network on a single batch of data
        Args:
            X: ndarray - input data
            y: ndarray - target data

        Returns:
            loss: float - train loss
            acc: float - train accuracy

        """
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        metrics = {metric.name: metric(y, y_pred) for metric in self.metrics}
        # Calculate the gradient of the loss function w.r.t. y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate
        self._backward_pass(loss_grad=loss_grad)

        return loss, metrics

    def fit(self, X, y, n_epochs, batch_size,
            validation_data=None,
            validation_split=0.0,
            early_stopping=False,
            patience=5
            ):
        """

        Args:
            X: ndarray - input data
            y: ndarray - target data
            n_epochs: int - number of epochs
            batch_size: int - number of samples per gradient update
            validation_data: (X_val, y_val) -
                data on which to evaluate the loss and any model metrics at the end of each epoch
            validation_split: float - fraction of training data to be used as validation data
            early_stopping: bool -
                whether to stop training early if validation loss doesn't improve after a given number of epochs
            patience: int - number of epochs to wait before stopping training

        Returns:

        """
        best_val_loss = np.inf
        patience_counter = 0

        if validation_data:
            val_set = {'X': validation_data[0], 'y': validation_data[1]}
        elif validation_data > 0.0:
            val_size = int(len(X) * validation_split)
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            val_indices, train_indices = indices[:val_size], indices[val_size:]
            val_set = {'X': X[val_indices], 'y': y[val_indices]}
            X, y = X[val_indices], y[val_indices]
        else:
            val_set = None

        # for epoch in self.progressbar(range(n_epochs)):
        for epoch in tqdm(range(n_epochs), desc='Training', unit='epoch'):
            batch_error = []
            train_metrics = {f'train_{metric.name}': [] for metric in self.metrics}
            val_metrics = {f'train_{metric.name}': [] for metric in self.metrics}
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, metrics = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
                for name, value in metrics.items():
                    train_metrics[f'train_{name}'].append(value)

            train_loss = np.mean(batch_error)
            train_metrics = {name: np.mean(train_metrics[name]) for name in self.metrics}
            self.errors["training"].append(train_loss)

            logs = {'train_loss': train_loss}
            logs.update(train_metrics)

            if val_set is not None:
                val_loss, val_metrics = self.test_on_batch(val_set["X"], val_set["y"])
                self.errors["validation"].append(val_loss)
                val_metrics = {f'val_{name}': value for name, value in val_metrics.items()}

                logs = {'val_loss': val_loss}
                logs.update(val_metrics)

                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch + 1}')
                        break

        return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, X, training=True):
        """
        Perform forward pass through the network
        Args:
            X: ndarray - input data
            training: True - whether the model is in training mode

        Returns: ndarray - output of the network

        """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        """
        Perform backward pass through the network
        Args:
            loss_grad: ndarray - gradient of the loss

        Returns:
        """
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def summary(self, name="Model Summary"):
        """
        Print a summary of the model's architecture and parameters
        Args:
            name: string - optional name for the summary

        Returns:
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
        Make predictions on the given input data
        Args:
            X: ndarray - input data

        Returns:

        """
        self._forward_pass(X, training=False)
