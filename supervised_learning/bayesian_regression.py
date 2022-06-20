

class BayesianRegression(object):
    """
    Bayesian regression model. Assumes Normal prior and likelihood for the weights and scaled inverse
    chi-squared prior and likelihood for the variance of the weights.
    """
    def __init__(self, n_draws, mu0, omega0, nu0, sigma_sq0, poly_degree=0, cred_int=95):
        """

        :param n_draws:
        :param mu0:
        :param omega0:
        :param nu0:
        :param sigma_sq0:
        :param poly_degree:
            If poly_degree is specified the features will be transformed to with a polynomial basis function,
            which allows for polynomial regression.

        :param cred_int:
        """
        ...

    def _draw_scaled_inv_chi_sq(self, n, df, scale):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X, eti=False):
        ...
