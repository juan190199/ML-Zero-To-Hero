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


class DiscreteDistribution(Distribution):
    def probability(self, value):
        raise NotImplementedError("Subclasses should override.")


class Multinomial(DiscreteDistribution):
    def __init__(self, categoryCounts, smoothingFactor=1.0):
        self.categoryCounts = categoryCounts
        self.numPoints = float(sum(categoryCounts.values()))
        self.numCategories = float(len(categoryCounts))
        self.smoothingFactor = float(smoothingFactor)

    def probability(self, value):
        if not value in self.categoryCounts:
            return 0.0
        numerator = float(self.categoryCounts[value]) + self.smoothingFactor
        denominator = self.numPoints + self.numCategories * self.smoothingFactor
        return numerator / denominator

    def __str__(self):
        return "Discrete Multinomial distribution: buckets = %s" % self.categoryCounts

    @classmethod
    def mleEstimate(cls, points):
        categoryCounts = collections.Counter()
        for point in points:
            categoryCounts[point] += 1
        return cls(categoryCounts)


class Binary(Multinomial):
    def __init__(self, trueCount, falseCount, smoothingFactor=1.0):
        categoryCounts = {True: trueCount, False: falseCount}
        Multinomial.__init__(self, categoryCounts, smoothingFactor)

    def __str__(self):
        return "Discrete Binary distribution: true count = %s, false count = %s" % (
        self.categoryCounts[True], self.categoryCounts[False])

    @classmethod
    def mleEstimate(cls, points, smoothingFactor=1.0):
        trueCount = 0
        for point in points:
            if point: trueCount += 1
        falseCount = len(points) - trueCount
        return cls(trueCount, falseCount, smoothingFactor)


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
