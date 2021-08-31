import numpy as np

from utils.data_manipulation import (Node, Tree)


class DensityTree(Tree):
    """
    Tree-like model (based on [1]) for estimating the probability density of the given data

    Reference
    ----------
    [1] Ram, Parikshit & Gray, Alexander. (2011). Density estimation trees.
    Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
    627-635. 10.1145/2020408.2020507.
    """

    def __init__(self):
        super(DensityTree, self).__init__()

    def make_density_split_node(node, n_samples, feature_indices):
        """
        Selects dimension and threshold where node is to be split up

        :param node: Node
            Node to be split

        :param n_samples: int
            Number of training instances

        :param feature_indices: ndarray of shape (D_try, )
            Contains feature indices to be considered in the present split

        :return: tuple
            Tuple of left and right children nodes (to be placed on the stack)

        """
        n_samples_node, n_features = node.data.shape
        m, M = node.box
        v = np.prod(M - m)
        if v <= 0:
            raise ValueError("Zero volume (should not happen)")

        # Find best feature j (among 'feature_indices') and best threshold t for the split
        e_min = float("inf")
        j_min, t_min = None, None

        for j in feature_indices:
            # Duplicate feature values have to be removed because candidate thresholds are
            # the midpoints of consecutive feature values, not the feature value itself
            dj = np.sort(np.unique(node.data[:, j]))
            # Compute candidate thresholds
            tj = (dj[1:] + dj[:-1]) / 2

            # Compute Leave-One-Out error of resulting children nodes for each candidate threshold
            for t in tj:
                # Compute number of instances in left and right children
                nl = np.sum(node.data[:, j] <= t)
                nr = n_samples_node - nl
                # Compute volume of left and right nodes
                vl = t / (M[j] - m[j])  # vl = v * t / (M[j] - m[j])
                vr = 1.0 - vl  # vr = v - vl
                # Notice actual volumes are commented. These differ by the constant factor v.

                if vl == 0 or vr == 0:
                    continue
                # Compute LOO errors
                el = (nl / (n_samples * vl)) * (nl / n_samples - 2.0 * ((nl - 1) / (n_samples_node - 1)))
                er = (nr / (n_samples * vr)) * (nr / n_samples - 2.0 * ((nr - 1) / (n_samples_node - 1)))

                # Choose best threshold that minimizes sum of LOO error
                loo_error = el + er
                if loo_error < e_min:
                    e_min = loo_error
                    j_min = j
                    t_min = t

        # Create children
        left = Node()
        right = Node()

        # Initialize 'left' and 'right' with the data subsets and bounding boxes
        # according to the optimal split found above
        left.data = node.data[node.data[:, j_min] <= t_min, :]
        left.box = m.copy(), M.copy()
        left.box[1][j_min] = t_min

        right.data = node.data[node.data[:, j_min] > t_min, :]
        right.box = m.copy(), M.copy()
        right.box[0][j_min] = t_min

        node.left = left
        node.right = right
        node.feature = j_min
        node.threshold = t_min

        return left, right

    def make_density_leaf_node(node, n_samples):
        """
        Compute and store leaf response

        :param node: Node
            Node that has reached termination criterion

        :param n_samples: int
            Number of training instances

        :return:
        """
        n_samples_node = node.data.shape[0]
        m, M = node.box
        v = np.prod(M - m)
        node.response = n_samples_node / (n_samples * v)

    def fit(self, data, prior, n_min=20):
        """

        :param data: ndarray of shape (n_samples, n_features)
            Training data

        :param prior: int
            Prior probability of target class

        :param n_min: int, default=20
            The minimum number of samples required to split an internal node

        :return:
        """
        self.prior = prior
        n_samples, n_features = data.shape
        n_possible_features = int(np.sqrt(n_features))  # Number of features to consider for each split decision

        # Find and remember the tree's bounding box,
        # i.e. the lower and upper limits of the training feature set
        m, M = np.min(data, axis=0), np.max(data, axis=0)
        self.box = m.copy(), M.copy()

        # Identify invalid features and adjust the bounding box
        # (If m[j] == M[j] for some j, the bounding box has zero volume,
        # causing divide-by-zero errors later on. These
        # features are excluded from splitting and adjust the bounding box limits
        # such that invalid features have no effect on the volume.)
        valid_features = np.where(m != M)[0]
        invalid_features = np.where(m == M)[0]
        M[invalid_features] = m[invalid_features] + 1

        # Initialize the root node
        self.root.data = data
        self.root.box = m.copy(), M.copy()

        # Build the tree
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n_samples_node = node.data.shape[0]  # Number of instances in present node
            if n_samples_node >= n_min:
                perm = np.random.permutation(len(valid_features))
                left, right = self.make_density_split_node(node, n_samples, valid_features[perm][:n_possible_features])
                stack.append(left)
                stack.append(right)
            else:
                self.make_density_leaf_node(node, n_samples)

    def predict(self, x):
        """
        Computes p(x | y) * p(y) if x is within the tree's bounding box. Otherwise return 0

        :param x: ndarray of shape (1, n_features)
            Sample point (test instance)

        :return:
            Return p(x | y) * p(y) if x is within the tree's bounding box. Otherwise return 0
        """
        m, M = self.box
        if np.any(x < m) or np.any(x > M):
            return 0.0
        else:
            return self.prior * self.find_leaf(x).response
