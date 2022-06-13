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
    """
    A Gaussian exponential family, used to fit a classical linear model.
    The conditional distribution of y|X is modeled as a Gaussian distribution
    """
    has_dispersion = True

    def inv_link(self, nu):
        return nu

    def d_inv_link(self, nu, mu):
        return np.ones(shape=nu.shape)

    def variance(self, mu):
        return np.ones(shape=mu.shape)

    def deviance(self, y, mu):
        return np.sum((y - mu) ** 2)

    def sample(self, mus, dispersion):
        return np.random.normal(mus, np.sqrt(dispersion))

    def initial_working_response(self, y):
        return y

    def initial_working_weights(self, y):
        return (1 / len(y)) * np.ones(len(y))


class Bernoulli(ExponentialFamily, ExponentialFamilyMixin):
    """
    A Bernoulli exponential family, used to fit a classical logistic model.
    The conditional distribution of y|X is modeled as a Bernoulli distribution
    """
    has_dispersion = False

    def inv_link(self, nu):
        return 1 / (1 + np.exp(-nu))

    def d_inv_link(self, nu, mu):
        return mu * (1 - mu)

    def variance(self, mu):
        return mu * (1 - mu)

    def deviance(self, y, mu):
        return -2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

    def sample(self, mus, dispersion):
        return np.random.binomial(1, mus)

    def initial_working_response(self, y):
        return (y - 0.5) / 0.25

    def initial_working_weights(self, y):
        return (1 / len(y)) * 0.25 * np.ones(len(y))


class QuasiPoisson(ExponentialFamily, ExponentialFamilyMixin):
    ...


class Poissson(QuasiPoisson):
    ...


class Gamma(ExponentialFamily, ExponentialFamilyMixin):
    ...


class Exponential(Gamma):
    ...
