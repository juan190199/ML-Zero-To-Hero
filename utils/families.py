from abc import ABCMeta, abstractmethod
import numpy as np


class ExponentialFamily(metaclass=ABCMeta):
    ...


class ExponentialFamilyMixin:
    ...


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
