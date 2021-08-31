import numpy as np

"""
Collection of activation functions
Reference: https://en.wikipedia.org/wiki/Activation_function
"""


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class TanH():
    ...


class ReLu():
    ...


class LeakyReLu():
    ...


class ELU():
    ...


class SELU():
    ...


class SoftPlus():
    ...
