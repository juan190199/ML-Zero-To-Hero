"""
Exponential families for GLM
"""
from abc import ABCMeta, abstractmethod
import numpy as np


class ExponentialFamily(metaclass=ABCMeta):
    @abstractmethod
    def inv_link(self, nu):
        """
        The inverse link function

        Parameters
        ----------
        nu:

        Returns
        -------

        """
        pass

    @abstractmethod
    def d_inv_link(self, nu, mu):
        """
        Derivative of the inverse link function

        Parameters
        ----------
        nu:

        mu:

        Returns
        -------

        """
        pass

    @abstractmethod
    def variance(self, mu):
        """
        The variance function linking the mean to the variance of the distribution

        Parameters
        ----------
        mu:

        Returns
        -------

        """
        pass

    @abstractmethod
    def deviance(self, y, mu):
        """
        The deviance of the family. Used as a measure of model fit.
        Parameters
        ----------
        y:
        mu:

        Returns
        -------

        """
        pass

    @abstractmethod
    def sample(self, mus, dispersion):
        """
        A sampler from the conditional distribution of the response
        Parameters
        ----------
        mus:
        dispersion:

        Returns
        -------

        """
        pass


class ExponentialFamilyMixin:
        """
        Implementation of methods common to all ExponentialFamily's
        """
        def penalized_deviance(self, y, mu, alpha, coef):
            return self.deviance(y, mu) + alpha * np.sum(coef[1:] ** 2)


class Gaussian(ExponentialFamily, ExponentialFamilyMixin):
        """
        A Gaussian exponential family, used to fit a classical linear model.

        The GLM fit with this family has the following structure equation:
            y | X ~ Gaussian(mu = X beta, sigma = dispersion)

        Here, sigma is a nuisance parameter
        """
        has_dispersion = True

        def inv_link(self, nu):
            return nu

        def d_inv_link(self, nu, mu):
            return np.ones(shape=nu.shape)

        def variance(self, mu):
            return np.ones(shape=mu.shape)

        def sample(self, mus, dispersion):
            return np.random.normal(mus, np.sqrt(dispersion))

        def initial_working_response(self, y):
            return y

        def initial_working_weights(self, y):
            return (1 / len(y)) * np.ones(len(y))


class Bernoulli(ExponentialFamily, ExponentialFamilyMixin):
    """
    A Bernoulli exponential family, used to fit a classical logistic model.

        The GLM fit with this family has the following structure equation:
            y | X ~ Bernoulli(p = X beta)
    """
    def inv_link(self, nu):
        return 1 / (1 + np.exp(-nu))

    def d_inv_link(self, nu, mu):
        return mu * (1 - mu)

    def variance(self, mu):
        return mu * (1 - mu)

    def deviance(self, y, mu):
        return -2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

    def sample(self, mus, dispersion):
        np.random.binomial(1, mus)

    def initial_working_response(self, y):
        return (y - 0.5) / 0.25

    def initial_working_weights(self, y):
        return (1 / len(y)) * 0.25 * np.ones(len(y))