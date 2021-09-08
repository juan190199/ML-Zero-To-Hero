import numpy as np
from utils.data_operation import calculate_covariance_matrix

class PCA():
    """
    A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and
    maximizing the variance along each feature axis.
    """

    def transform(self, X, n_components):
        """
        Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset

        :param X: ndarray of shape (n_samples, n_features)
            Design matrix

        :param n_components: int
            Number of principal components

        :return: ndarray of shape (n_samples, n_components)
            Transformed dataset
        """

        covariance_matrix = calculate_covariance_matrix(X)

        evalues, evectors = np.linalg.eig(covariance_matrix)

        # Sort evalues and corresponding evectors from largest
        # to smallest and select the first n_components
        idx = evalues.argsort()[::-1]
        evalues = evalues[idx][:n_components]
        evectors = np.atleast_1d(evectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(evectors)

        return X_transformed
