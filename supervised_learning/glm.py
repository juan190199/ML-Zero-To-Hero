import numpy as np
import scipy.stats as sts
import patsy as pt

from utils.sanity_checks import (check_types, check_commensurate, check_intercept, check_offset, check_sample_weights,
                                 has_converged, default_X_names, default_y_name)


class GLM:
    """
    A Generalized Linear Model is a generalization of the classical linear and logistic models to other conditional
    distributions of response y. A GLM is specified by a *link function* G and a family of *conditional distributions*
    dist, with the model specification given by

        y|X ~ dist(eta = G(X * theta))

    Here theta are the parameters fit in the model, with X * theta a matrix multiplication just like in linear
    regression. Above, eta is a *parameter* of the one parameter family of distributions dist.

    In this implementation, a specific GLM is specified with a *family* object of ExponentialFamily type, which
    contains the information about the conditional distribution of y, and its connection to X, needed to construct
    the model.

    The model is fit to data using the well known Fisher Scoring algorithm, which is a version of Newton's method
    where the Hessian is replaced with its expectation w.r.t. the assumed distribution of y
    """

    def __init__(self, family, alpha=0.0):
        """

        :param family:
        :param alpha:
        """
        self.family = family
        self.alpha = alpha
        self.formula = None
        self.X_info = None
        self.X_names = None
        self.y_name = None
        self.coef_ = None
        self.deviance_ = None
        self.n = None
        self.p = None
        self.information_matrix_ = None


    def fit(self, X, y=None, formula=None, *, X_names=None, y_name=None, **kwargs):
        """

        :param X:
        :param y:
        :param formula:
        :param X_names:
        :param y_name:
        :param kwargs:
        :return:
        """
