import warnings

import numpy as np
from scipy import linalg

from utils.data_operation import logsumexp


class LDA(object):
    """
    Linear Discriminant Analysis: Classifier with a linear decision boundary,
    generated by fitting class conditional densities to the data using Bayes' rule

    The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix

    The fitted model can also be used to reduce the dimensionality of the input,
    by projecting it to the most discriminative directions.

    Attributes
    -----------
    * coef_: array-like of shape (rank, n_classes - 1)
        Coefficients of the features in the linear decision function.
        rank is min(rank_features, n_classes) where rank_features is the dimensionality
        of the spaces spanned by the features (i.e. n_features excluding redundant features).

    * covariance_: array-like of shape (n_features, n_features)
        Covariance matrix (shared by all classes)

    * means_: array-like of shape (n_classes, n_features)
        Class-wise means

    * priors_: array-like of shape (n_classes,)
        Class priors (sum to 1)

    * scalings_: array-like of shape (rank, n_classes - 1)
        Scaling of the features in the space spanned by the class centroids.

    * xbar_: array-like of shape (n_features,)
        Overall mean
    """

    def __init__(self, priors=None, n_components=None, store_covariance=False, tol=1.0e-4):
        """

        :param priors: array-like of shape (n_classes,), default=None
            Priors on classes

        :param n_components: int, default=None
            Number of components (< n_classes - 1) for dimensionality reduction

        :param store_covariance: bool, default=None
            If True the covariance matrix is computed and stored in 'self.covariance_' attribute

        :param tol:
        """
        self.priors = np.asarray(priors) if priors is not None else None
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol

    def fit(self, X, y, store_covariance=False):
        """
        Fit LDA model according to the given training data and parameters.

        :param X: array-like of shape (n_samples, n_features)
            Training data

        :param y: array-like of shape (n_samples,) or (n_samples, n_classes)
            Target data

        :param store_covariance: bool, default=None
            If True the covariance matrix is computed and stored in 'self.covariance_' attribute

        :return:
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        if self.priors is None:
            _, y_t = np.unique(y, return_inverse=True)
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError("n_components cannot be larger than min(n_classes - 1, n_features)")
            self._max_components = self.n_components

        means = []
        Xc = []
        cov = None
        if store_covariance:
            cov = np.empty((n_features, n_features))

        # Normalize data
        for idx in range(n_classes):
            Xg = X[y == idx]
            meang = Xg.mean(axis=0)  # Group mean
            means.append(meang)
            # Centered group data
            Xgc = Xg - meang
            Xc.append(Xgc)
            if store_covariance:
                cov += np.dot(Xgc.T, Xgc)

        # The covariance matrix is a weighted average of the local covariance matrix estimates within each class
        if store_covariance:
            cov /= (n_samples - n_classes)
            self.covariance_ = cov

        self.means_ = np.asarray(means)
        Xc = np.concatenate(Xc, axis=0)

        # 1. Within (univariate) scaling by with classes std-dev
        std = Xc.std(axis=0)
        # Sanity check division by zero
        std[std == 0] = 1
        fac = 1. / (n_samples - n_classes)

        # 2. Within variance scaling
        X = np.sqrt(fac) * (Xc / std)
        # SVD of centered (within) scaled
        U, S, Vt = linalg.svd(X, full_matrices=False)

        rank = np.sum(S > self.tol)
        if rank < n_features:
            warnings.warn("Variables are collinear.")
        # Scaling of within covariance is: V' 1/S
        scalings = (Vt[:rank] / std).T / S[:rank]

        # 3. Between variance scaling
        # Overall mean
        self.xbar_ = np.dot(self.priors_, self.means_)
        # Scale weighted centers
        X = np.dot(((np.sqrt((n_samples * self.priors_) * fac)) * (self.means_ - self.xbar_).T).T, scalings)
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the (n_classes) centers
        _, S, Vt = linalg.svd(X, full_matrices=0)

        rank = np.sum(S > self.tol * S[0])
        self.scalings_ = np.dot(scalings, Vt.T[:, :rank])
        self.coef_ = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = (-0.5 * np.sum(self.coef_ ** 2, axis=1) + np.log(self.priors_))

        return self

    def _decision_function(self, X):
        """
        Computes values related to each class, per sample.

        :param X:array-like of shape (n_samples, n_features)
            Test data

        :return: array-like of shape (n_samples, n_classes) or (n_samples,)
            Decision function values related to each class, per sample.
        """
        # Center and scale data
        X = np.dot(X - self.xbar_, self.scalings_)
        return np.dot(X, self.coef_.T) + self.intercept_

    def decision_function(self, X):
        """

        :param X:array-like of shape (n_samples, n_features)
            Test data

        :return: array-like of shape (n_samples, n_classes) or (n_samples,)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,], giving the log likelihood ratio of the positive class.
        """
        dec_function = self._decision_function(X)
        if len(self.classes_) == 2:
            return dec_function[:, 1] - dec_function[:, 0]
        return dec_function

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        :param X: array-like of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples,)
            Vector of predicted labels for each sample
        """
        d = self._decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        return y_pred

    def predict_proba(self, X):
        """
        Estimate probability

        :param X_test: array-like of shape (n_samples, n_features) -- Input data
        :return: ndarray of shape (n_samples, n_classes) -- Estimated probabilities
        """
        values = self._decision_function(X)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        """
        Estimate log-probability

        :param X_test: array-like of shape (n_samples, n_features) -- Input data
        :return: ndarray of shape (n_samples, n_features) -- Estimated log-probabilities
        """
        values = self._decision_function(X)
        loglikelihood = (values - values.max(axis=1)[:, np.newaxis])
        normalization = logsumexp(loglikelihood, axis=1)
        return loglikelihood - normalization[:, np.newaxis]