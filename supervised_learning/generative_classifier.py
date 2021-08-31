import numpy as np

from density_tree import DensityTree
from density_forest import DensityForest


class GenerativeClassifier:
    """
    Combines 10 DensityTree objects.
    One object of DensityTree is to be trained with only one class of data from the training data.

    To train the generative classifier with the full data set, it is needed to separate the data into 10 subsets
    and then train 10 DensityTree objects, each trained for a different subset.

    For prediction, all 10 DensityTree returns the prediction p(x|y) * p(y) for a data point.
    The target for the DensityTree maximizing p(x|y) * p(y) is the prediction of the generative classifier

    Attributes
    -----------
    * trees_: List
        List containing 10 DensityTree objects to train each of them with only one class of data from the training data.
    """
    def __init__(self):
        self.trees_ = [DensityTree() for i in range(10)]

    def fit(self, data, target, n_min=20):
        """

        :param data: array-like of shape (n_instances, n_features)
            Design matrix with data

        :param target: array-like of shape (n_instances, )
            Ground-truth responses

        :param n_min: int, default=20
            The minimum number of samples required to split an internal node

        :return:
        """
        data_subsets = [data[target == i] for i in range(10)]
        N = len(target)
        for i, tree in enumerate(self.trees_):
            tree.fit(data_subsets[i], len(data_subsets[i]) / N, n_min)

    def predict(self, x):
        """

        :param x: array-like of shape (1, n_features)
            Sample point (test instance)

        :return: int
            Predicted target for sample point x
        """
        return np.argmax([tree.predict(x) for tree in self.trees_])


class GenerativeClassifierDensityForest:
    """
    Generative Classifier Density Forest
    """
    def __init__(self, n_trees=100):
        """

        :param n_trees: int, default=100
            The number of trees in the forest.
        """
        self.forests_ = [DensityForest(n_trees) for i in range(10)]

    def fit(self, data, target, n_min=20):
        """

        :param data: array-like of shape (n_instances, n_features)
            Design matrix

        :param target:

        :param n_min: int, default=20
            The minimum number of samples required to split an internal node.

        :return:
        """
        data_subsets = [data[target == i] for i in range(10)]
        N = len(target)
        for i, forest in enumerate(self.forests_):
            forest.fit(data_subsets[i], len(data_subsets[i]) / N, n_min)

    def predict(self, x):
        """

        :param x: array-like of shape (1, n_features)
            Sample point (test instance)

        :return: int
            Return target maximizing the posterior probability p(x | y) * p(y)
        """
        return np.argmax([forest.predict(x) for forest in self.forests_])
