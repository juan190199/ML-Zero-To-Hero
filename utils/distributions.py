import os
import sys
import math
import collections


class Distribution(object):
    def __init__(self):
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def mleEstimate(cls, points):
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def momEstimate(cls, points):
        raise NotImplementedError("Subclasses should override.")


class ContinuousDistribution(Distribution):
    def pdf(self, value):
        raise NotImplementedError("Subclasses should override.")

    def cdf(self, value):
        raise NotImplementedError("Subclasses should override.")


class Gaussian(ContinuousDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        if std == 0.0:
            raise ParametrizationError("Standard deviation must be non-zero")
        if std < 0.0:
            raise ParametrizationError("Standard deviation must be positive")
        self.variance = math.pow(std, 2.0)

    def pdf(self, value):
        numerator = math.exp(-math.pow(float(value - self.mean) / self.std, 2.0) / 2.0)
        denominator = math.sqrt(2 * math.pi * self.variance)
        return numerator / denominator

    def cdf(self, value):
        return 0.5 * (1.0 + math.erf((value - self.mean) / math.sqrt(2.0 * self.variance)))

    def __str__(self):
        return "Continuous Gaussian (Normal) distribution: mean = %s, standard deviation = %s" % (self.mean, self.std)

    @classmethod
    def mleEstimate(cls, points):
        numPoints = float(len(points))
        if numPoints <= 1:
            raise EstimationError("Must provide at least 2 training points")

        mean = sum(points) / numPoints

        variance = 0.0
        for point in points:
            variance += math.pow(float(point) - mean, 2.0)
        variance /= (numPoints - 1.0)
        std = math.sqrt(variance)

        return cls(mean, std)




##########   Errors   ##########

class EstimationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ParametrizationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
