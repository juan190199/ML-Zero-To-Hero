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


class Dense(Layer):
    """
    A fully-connected NN layer
    """

    def __init__(self, n_units, input_shape=None):
        """

        :param n_units: int
            Number of neurons in the layer

        :param input_shape: tuple
            Expected input shape of the layer.
            For dense layers, a single digit indicating the number of features of the input
            must be specified if it is the first layer in the network
        """
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.w = None
        self.b = None

    def initialize(self, optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input.shape[0])
        self.w = np.random.uniform(-limit, limit, (self.input.shape[0], self.n_units))
        self.b = np.zeros((1, self.n_units))
        # Weight optimizers
        self.w_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)

    def forward_pass(self, X, training):
        self.layer_input = X
        return X.dot(self.w) + self.b

    def backward_pass(self, accum_grad):
        # Save weights used during forward pass
        w = self.w

        if self.trainable:
            # Calculate gradient w.r.t. layer weights
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_b = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layers weight
            # w = w - self.learning_rate * grad_w
            self.w = self.w_opt.update(self.w, grad_w)
            self.b = self.b_opt.update(self.b, grad_b)

        # Return accumulated gradient for next_layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(w.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units,)


class Dropout(Layer):
    """
    A layer that randomly sets a fraction p of the output units of the previous layer to zero
    """

    def __init__(self, p=0.2):
        """

        :param p: float
            Probability that unit x is set to zero.
        """
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X, training):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
            return X * c

    def backward_pass(self, accum_grad):
        return accum_grad * self._mask

    def output_shape(self):
        return self.input_shape


activation_functions = {
    'relu': ReLu,
    'sigmoid': Sigmoid,
    'selu': SELU,
    'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLu,
    'tanh': TanH,
    'softplus': SoftPlus
}


class Activation(Layer):
    """
    A layer that applies an activation to the input
    """

    def __init__(self, name):
        """

        :param name: string
            Name of the activation function that will be used.
        """
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Conv2D(Layer):
    """
    A 2D Convolution layer
    """

    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1):
        """

        :param n_filters: int
            Number of filters that will convolve the input matrix.
            The number of channels of the output shape.

        :param filter_shape: tuple
            (filter_height, filter_width)

        :param input_shape: tuple
            Shape of the expected input of the layer (batch_size, channels, height, width)
            Only needs to be specified for the first layer in the network

        :param padding: string
            Possible options are: 'same' or 'valid'.
            'same' results in padding being added so that the output height and width
            matches the input height and width.
            For 'valid', no padding is added.

        :param stride: int
            The stride length of the filters during the convolution over the input
        """
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True

    def initialize(self, optimizer):
        # Initialize the weights
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / np.sqrt(np.prod(self.filter_shape))
        self.w = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.b = np.zeros((self.n_filters, 1))
        # Weight optimizers
        self.w_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)

    def forward_pass(self, X, training):
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        # Turn image shape into column shape (enables dot product between input and weights)
        self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.w_col = self.w.reshape((self.n_filters, -1))
        # Calculate output
        output = self.w_col.dot(self.X_col) + self.b
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size,))
        # Redistribute axes so that batch size comes first
        return output.transpose(3, 0, 1, 2)

    def backward_pass(self, accum_grad):
        # Reshape accumulated gradient into column shape
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        if self.trainable:
            # Take dot product between column shaped accum. gradient and
            # column shape layer input to determine the gradient at the layer w.r.t. layer weights
            grad_w = accum_grad.dot(self.X_col.T).reshape(self.w.shape)
            # The gradient w.r.t. bias terms is the sum similarly as in the Dense layer
            grad_b = np.sum(accum_grad, axis=1, keepdims=True)

            # Update the layer weights
            self.w = self.w_opt.update(self.w, grad_w)
            self.b = self.b_opt.update(self.b, grad_b)

        # Recalculate the gradient which will propagated back to previous layer
        accum_grad = self.w_col.T.dot(accum_grad)
        # Reshape from column shape to image shape
        accum_grad = column_to_image(accum_grad,
                                     self.layer_input.shape,
                                     self.filter_shape,
                                     stride=self.stride,
                                     output_shape=self.padding)

        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)


class PoolingLayer(Layer):
    """
    Parent class of MaxPooling2D and AveragePooling2D
    """

    def __init__(self, pool_shape=(2, 2), stride=1, padding=0):
        """

        :param pool_shape:
        :param stride:
        :param padding:
        """
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True

    def forward_pass(self, X, training=True):
        """

        :param X:
        :param training:
        :return:
        """
        self.layer_input = X

        batch_size, channels, height, width = X.shape

        _, out_height, out_width = self.output_shape()

        X = X.reshape(batch_size * channels, 1, height, width)
        X_col = image_to_column(X, self.pool_shape, self.stride, self.padding)

        # MaxPool or AveragePool specific method
        output = self._pool_forward(X_col)

        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)

        return output

    def backward_pass(self, accum_grad):
        batch_size, _, _, _ = accum_grad.shape
        channels, height, width = self.input_shape
        accum_grad = accum_grad.transpose(2, 3, 0, 1).ravel()

        # MaxPool or AveragePool specific method
        accum_grad_col = self._pool_backward(accum_grad)

        accum_grad = column_to_image(accum_grad_col,
                                     (batch_size * channels, 1, height, width),
                                     self.pool_shape,
                                     self.stride,
                                     0)

        accum_grad = accum_grad.reshape((batch_size,) + self.input_shape)

        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) / self.stride + 1
        out_width = (width - self.pool_shape[1]) / self.stride + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)


class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        arg_max = np.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, accum_grad):
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        arg_max = self.cache
        accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad
        return accum_grad_col


class AveragePooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        output = np.mean(X_col, axis=0)
        return output

    def _pool_backward(self, accum_grad):
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        accum_grad_col[:, range(accum_grad.size)] = 1. / accum_grad_col.shape[0] * accum_grad
        return accum_grad_col


class ConstantPadding2D(Layer):
    """
    Adds rows and columns of constant values to the input.
    Expects the input to be of shape (batch_size, channels, height, width)
    """

    def __init__(self, padding, padding_value=0):
        self.padding = padding
        self.trainable = True
        if not isinstance(padding[0], tuple):
            self.padding = ((padding[0], padding[0]), padding[1])
        if not isinstance(padding[1], tuple):
            self.padding = (padding[0], (padding[1], padding[1]))
        self.padding_value = padding_value

    def forward_pass(self, X, training):
        output = np.pad(
            X,
            pad_width=((0, 0), (0, 0), self.padding[0], self.padding[1]),
            mode='constant',
            constant_values=self.padding_value
        )
        return output

    def backward_pass(self, accum_grad):
        pad_top, pad_left = self.padding[0][0], self.padding[1][0]
        height, width = self.input_shape[1], self.input_shape[2]
        accum_grad = accum_grad[:, :, pad_top:pad_top + height, pad_left:pad_left + width]
        return accum_grad

    def output_shape(self):
        new_height = self.input_shape[1] + np.sum(self.padding[0])
        new_width = self.input_shape[2] + np.sum(self.padding[1])
        return (self.input_shape[0], new_height, new_width)




########################################################################################################################
########################################################################################################################

def determine_padding(filter_shape, output_shape='same'):
    """
    Method which calculates the padding based on the specified output shape and the shape of the filters

    :param filter_shape:
    :param output_shape:
    :return:
    """
    # No padding
    if output_shape == 'valid':
        return (0, 0), (0, 0)
    # Pad so that the output is the same as input shape (given that stride=1)
    elif output_shape == 'same':
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case, output_height = height and stride = 1.
        # This gives the expression for the padding below
        pad_h1 = int(math.floor((filter_height - 1) / 2))
        pad_h2 = int(math.ceil((filter_height - 1) / 2))
        pad_w1 = int(math.floor((filter_width - 1) / 2))
        pad_w2 = int(math.ceil((filter_width - 1) / 2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)


# Reference: CS231n Stanford university
def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    """
    Calculate indices where the dot products are to be applied between weights and the image

    :param images_shape: Tuple
        (batch_size, channels, height, width)

    :param filter_shape: Tuple
        (height, width)

    :param padding: Tuple
        (pad_h1, pad_h2), (pad_w1, pad_w2)

    :param stride: int
        The stride length of the filters during the convolution over the input

    :return:
    """
    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    # Matrix of shape (size of filter, row index center of patch)
    # Size of filter is the multiplication of the filter dimensions
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    # Matrix of shape (size of filter, column index center of patch)
    # Size of filter is the multiplication of the filter dimensions
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    # Channel indices
    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    return (k, i, j)


# Reference: CS231n Stanford university
def image_to_column(images, filter_shape, stride, output_shape='same'):
    """
    Turns the image shaped input to column shape. Used during forward pass

    :param images: ndarray of shape (batch_size, channels, height, width)
        Data to be processed

    :param filter_shape: ndarray of shape (height, width)
        Filter to apply convolution to given image

    :param stride: int
        The stride length of the filters during the convolution over the input

    :param output_shape: ndarray of shape (n_filters, channels, height, width)
        Output after convolution

    :return:
    """
    filter_height, filter_width = filter_shape

    pad_h, pad_w = determine_padding(filter_shape, output_shape)

    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    # Calculate the indices where the dot products are to be applied between weights and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    # For all padded images, get for all channels the values of the 9 spaces
    # indexed by the coordinates (i, j) of the center of the filter.
    # images_padded[0, 0, i, j] denotes for the image 0 and channel 0,
    # all local values (in columns) of the padded image through all transitions
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


# Reference: CS231n Stanford university
def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    """
    Turns the column shaped input to image shape. Used during the backward pass

    :param cols:
    :param image_shape:
    :param filter_shape:
    :param stride:
    :param output_shape:
    :return:
    """
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.zeros((batch_size, channels, height_padded, width_padded))

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    # Return image without padding
    return images_padded[:, :, pad_h[0]:height + pad_h[0], pad_w[0]:width + pad_w[0]]
