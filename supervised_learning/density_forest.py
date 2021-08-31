import numpy as np

from supervised_learning.density_tree import DensityTree


class DensityForest:
    """
    Density Forest: Family of tree-like models (based on [1]) for estimating the probability density of the given data

    Reference
    ----------
    [1] Ram, Parikshit & Gray, Alexander. (2011). Density estimation trees.
    Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
    627-635. 10.1145/2020408.2020507.
    """

    def __init__(self, n_trees=100):
        """

        :param n_trees: int, default=100
            The number of trees in the forest.
        """
        self.trees_ = [DensityTree() for i in range(n_trees)]

    def fit(self, data, prior, n_min=20):
        """

        :param data: ndarray of shape (n_instances, n_features)
            Training data

        :param prior: int
            Prior probability of target class

        :param n_min: int, default=20
            The minimum number of samples required to split an internal node

        :return:
        """
        for tree in self.trees_:
            bootstrap_data = data[np.random.choice(len(data), size=len(data))]
            tree.fit(bootstrap_data, prior, n_min)

    def predict(self, x):
        """
        Compute mean mean of posterior probabilities of each tree in self.trees_

        :param x: ndarray of shape (1, n_features)
            Sample point (test instance)

        :return: int
            Return mean of posterior probabilities of each tree in self.trees_
        """
        return np.mean([tree.predict(x) for tree in self.trees_])
