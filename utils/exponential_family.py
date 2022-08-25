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