import math
import numpy as np

import copy

from deep_learning.activation_functions import (Sigmoid, ReLu, SoftPlus, LeakyReLu, TanH, ELU, SELU, Softmax)


class Layer(object):

    def set_input_shape(self, shape):
        """
        Sets the shape that the layer expects of the input in the forward pass method

        :param shape: tuple
            Expected shape of the input in the forward pass method

        :return:
        """
        self.input_shape = shape

    def layer_name(self):
        """
        The name of the layer. Used in model summary.
        :return:
        """
        return self.__class__.__name__

    def parameters(self):
        """
        Number of trainable parameters used by the layer
        :return:
        """
        return 0

    def forward_pass(self, X, training):
        """
        Propagates the signal forward in the network

        :param X:
        :param training:
        :return:
        """
        raise NotImplementedError

    def backward_pass(self, accum_grad):
        """
        Propogates the accumulated gradient backwards in the network.
        If it has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer.

        :param accum_grad:
        :return:
        """
        raise NotImplementedError

    def output_shape(self):
        """
        Shape of the output produced by forward pass
        :return:
        """
        raise NotImplementedError
