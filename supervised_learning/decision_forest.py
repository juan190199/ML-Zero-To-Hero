import numpy as np

from supervised_learning.decision_tree import DecisionTree


class DecisionForest:
    """
    Decision Forest: Family of tree-like model that predicts the value of a target variable
    based on several input variables.
    """

    def __init__(self, n_trees=100):
        """

        :param n_trees: int, default=100
            The number of trees in the forest.
        """
        self.trees_ = [DecisionTree() for i in range(n_trees)]

    def fit(self, data, labels, n_min=0):
        """

        :param data: ndarray of shape (n_instances, n_features)
            Training data

        :param labels:

        :param n_min: int, default=0
            The minimum number of samples required to split an internal node

        :return:
        """
        for tree in self.trees_:
            bootstrap = np.random.choice(len(data), size=len(data))
            tree.fit(data[bootstrap], labels[bootstrap], n_min)

    def predict(self, x):
        """

        :param x: ndarray of shape (1, n_features)
            Sample point (test instance)

        :return: int
            Return mean of posterior probabilities of each tree in self.trees_
        """
        return np.mean([tree.predict(x) for tree in self.trees_], axis=0)
