import collections
from math import log

from utils import distributions


##### Feature #####

class Feature(object):

    def __init__(self, name, distribution, value):
        self.name = name
        self.distribution = distribution
        self.value = value

    def __repr__(self):
        return self.name + " => " + str(self.value)

    def hashable(self):
        return (self.name, self.value)

    @classmethod
    def binary(cls, name):
        return cls(name, distributions.Binary, True)


##### ExtractedFeature #####

class ExtractedFeature(Feature):

    def __init__(self, object):
        name = self.__class__.__name__
        distribution = self.distribution()
        value = self.extract(object)
        super(ExtractedFeature, self).__init__(name, distribution, value)

    def extract(self, object):
        # returns feature value corresponding to |object|
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def distribution(cls):
        # returns the distribution this feature conforms to
        raise NotImplementedError("Subclasses should override.")


