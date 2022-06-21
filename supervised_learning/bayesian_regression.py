from __future__ import print_function, division
import numpy as np
from scipy.stats import chi2, multivariate_normal
from utils.data_manipulation import train_test_split, polynomial_features
from utils.data_operation import mean_squared_error


class BayesianRegression(object):

    def __init__(self, n_draws, mu0, omega0, nu0, sigma_sq0, poly_degree=0, cred_int=95):
        """

        :param n_draws: float
            The number of simulated draws from the posterior of the parameters

        :param mu0: ndarray
            The mean values of the prior normal distribution of the parameters

        :param omega0: ndarray
            The precision matrix of the prior normal distribution of the parameters (inverse variance matrix)

        :param nu0: float
            The degrees of freedom of the prior scaled inverse chi squared distribution

        :param sigma_sq0: float
            The scale parameter of the prior scaled inverse chi squared distribution.

        :param poly_degree: int
            The polynomial degree that the features should be transformed to. Allows for polynomial regression.

        :param cred_int: float
            The credibility interval (ETI in this impl.).
            95 => 95% credibility interval of the posterior of the parameters.
        """
        self.w = None
        self.n_draws = n_draws
        self.poly_degree = poly_degree
        self.cred_int = cred_int

        # Prior parameters
        self.mu0 = mu0
        self.omega0 = omega0
        self.nu0 = nu0
        self.sigma_sq0 = sigma_sq0

