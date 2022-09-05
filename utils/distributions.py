import os
import sys
import math
import collections


class Distribution(object):
    def __init__(self):
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def ml_estimator(cls, points):
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def mom_estimator(cls, points):
        raise NotImplementedError("Subclasses should override.")


class ContinuousDistribution(Distribution):
    def pdf(self, value):
        raise NotImplementedError("Subclasses should override.")

    def cdf(self, value):
        raise NotImplementedError("Subclasses should override.")


class Uniform(ContinuousDistribution):
    def __init__(self, alpha, beta):
        if alpha == beta: raise ParametrizationError("Alpha and beta cannot be equal")
        self.alpha = alpha
        self.beta = beta
        self.range = beta - alpha

    def pdf(self, value):
        if value < self.alpha or value > self.beta: return 0.0
        else: return 1.0 / self.range

    def cdf(self, value):
        if value < self.alpha: return 0.0
        elif value > self.beta: return 0.0
        else: return (value - self.alpha) / self.range

    def __str__(self, value):
        return "Continuous Uniform distribution: alpha = %s, beta = %s" % (self.alpha, self.beta)

    @classmethod
    def ml_estimator(cls, points):
        return cls(min(points), max(points))


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
    def ml_estimator(cls, points):
        n_points = float(len(points))
        if n_points <= 1:
            raise EstimationError("Must provide at least 2 training points")

        mean = sum(points) / n_points

        variance = 0.0
        for point in points:
            variance += math.pow(float(point) - mean, 2.0)
        variance /= (n_points - 1.0)
        std = math.sqrt(variance)

        return cls(mean, std)


class DiscreteDistribution(Distribution):
    def probability(self, value):
        raise NotImplementedError("Subclasses should override.")


class Multinomial(DiscreteDistribution):
    def __init__(self, category_counts, smoothing_factor=1.0):
        self.category_counts = category_counts
        self.n_points = float(sum(category_counts.values()))
        self.num_categories = float(len(category_counts))
        self.smoothing_factor = float(smoothing_factor)

    def probability(self, value):
        if not value in self.category_counts:
            return 0.0
        numerator = float(self.category_counts[value]) + self.smoothing_factor
        denominator = self.n_points + self.num_categories * self.smoothing_factor
        return numerator / denominator

    def __str__(self):
        return "Discrete Multinomial distribution: buckets = %s" % self.category_counts

    @classmethod
    def ml_estimator(cls, points):
        category_counts = collections.Counter()
        for point in points:
            category_counts[point] += 1
        return cls(category_counts)


class Binary(Multinomial):
    def __init__(self, true_count, false_count, smoothing_factor=1.0):
        category_counts = {True: true_count, False: false_count}
        Multinomial.__init__(self, category_counts, smoothing_factor)

    def __str__(self):
        return "Discrete Binary distribution: true count = %s, false count = %s" % (
        self.category_counts[True], self.category_counts[False])

    @classmethod
    def ml_estimator(cls, points, smoothing_factor=1.0):
        true_count = 0
        for point in points:
            if point:
                true_count += 1
        false_count = len(points) - true_count
        return cls(true_count, false_count, smoothing_factor)


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