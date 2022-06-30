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

    def _draw_scaled_inv_chi_sq(self, n, df, scale):
        X = chi2.rvs(size=n, df=df)
        sigma_sq = df * scale / X
        return sigma_sq

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        # If polynomial transformation
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        n_samples, n_features = np.shape(X)

        X_X = X.T.dot(X)

        # Least squares approximate of theta
        beta_hat = np.linalg.pinv(X_X).dot(X.T).dot(y)

        # The posteriors can be computed analytically since we assume conjugate priors for the likelihood.
        # Normal prior / likelihood => Normal posterior
        omega_n = X_X + self.omega0
        mu_n = np.linalg.pinv(omega_n).dot(X_X.dot(beta_hat) + self.omega0.dot(self.mu0))
        # Scaled inverse chi-squared prior / likelihood => Scaled inverse chi-squared posterior
        nu_n = self.nu0 + n_samples
        sigma_sq_n = (1.0 / nu_n) * (self.nu0 * self.sigma_sq0 + (
                y.T.dot(y) + self.mu0.T.dot(self.omega0).dot(self.mu0) - mu_n.T.dot(omega_n.dot(mu_n))))

        # Simulate parameter values for n_draws
        beta_draws = np.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = self._draw_scaled_inv_chi_sq(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:, 0], cov=sigma_sq * np.linalg.pinv(omega_n))
            # Save parameter draws
            beta_draws[i, :] = beta

        # Select the mean of the simulated variables as the ones used to make predictions
        self.w = np.mean(beta_draws, axis=0)

        # Lower and upper boundary of the credible interval
        l_eti = 50 - self.cred_int / 2
        u_eti = 50 + self.cred_int / 2
        self.eti = np.array([[np.percentile(beta_draws[:, i], q=l_eti), np.percentile(beta_draws[:, i], q=u_eti)] \
                             for i in range(n_features)])

    def predict(self, X, eti=False):

        # If polynomial transformation
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        y_pred = X.dot(self.w)
        # If the lower and upper boundaries for the 95%
        # equal tail interval should be returned
        if eti:
            lower_w = self.eti[:, 0]
            upper_w = self.eti[:, 1]
            y_lower_pred = X.dot(lower_w)
            y_upper_pred = X.dot(upper_w)
            return y_pred, y_lower_pred, y_upper_pred

        return y_pred
