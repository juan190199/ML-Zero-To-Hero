from abc import ABCMeta, abstractmethod
import numpy as np


class ExponentialFamily(metaclass=ABCMeta):
    """
    An exponential family must implement at least four methods and define one attribute
    """
    @abstractmethod
    def inv_link(self, nu):
        pass

    @abstractmethod
    def d_inv_link(self, nu, mu):
        pass

    @abstractmethod
    def variance(self, mu):
        pass

    @abstractmethod
    def deviance(self, y, mu):
        pass

    @abstractmethod
    def sample(self, mus, dispersion):
        pass


class ExponentialFamilyMixin:
    """
    Implementation of methods common to all ExponentialFamily objects
    """
    def penalized_deviance(self, y, mu, alpha, coef):
        return self.deviance(y, mu) + alpha * np.sum(coef[1:] ** 2)


class Gaussian(ExponentialFamily, ExponentialFamilyMixin):
    ...


class Bernoulli(ExponentialFamily, ExponentialFamilyMixin):
    ...


class QuasiPoisson(ExponentialFamily, ExponentialFamilyMixin):
    ...


class Poissson(QuasiPoisson):
    ...


class Gamma(ExponentialFamily, ExponentialFamilyMixin):
    ...


class Exponential(Gamma):
    ...
