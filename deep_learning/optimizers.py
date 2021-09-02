import numpy as np


class StochasticGradientDescent():
    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * grad_wrt_w
        return w - self.learning_rate * self.w_updt


class NesterovAcceleratedGradient():
    def __init__(self, learning_rate=0.001, momentum=0.4):
        ...

    def update(self, w, grad_wrt_w):
        ...


class Adagrad():
    def __init__(self, learning_rate=0.01):
        ...

    def update(self, w, grad_wrt_w):
        ...


class Adadelta():
    def __init__(self, rho=0.95, eps=1e-6):
        ...

    def update(self, w, grad_wrt_w):
        ...


class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        ...

    def update(self, w, grad_wrt_w):
        ...


class Adam():
    def __init__(self, learning_rate=0.01, b1=0.9, b2=0.999):
        ...

    def update(self, w, grad_wrt_w):
        ...
