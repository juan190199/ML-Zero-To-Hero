import math
import numpy as np


class NaiveBayes():
    """
    The Gaussian Naive Bayes classifier.
    """
    def fit(self, X, y):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Training data

        :param y: ndarray of shape (n_samples, )
            Target data

        :return: self
        """
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        # Calculate the mean and the variance of each feature for each class
        for i, c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            # Add the mean and variance for each feature
            for col in X_where_c.T:
                parameters = {'mean': col.mean(),
                              'var': col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        """
        Gaussian likelihood of the data x given mean and var

        :param mean:


        :param var:

        :param x:
        :return:
        """
        eps = 1e-4
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self, c):
        """
        Calculate prior of class c

        :param c: int
            Target class

        :return:
            Prior (number of samples where class == c / total number of samples)
        """
        frequency = np.mean(self.y == c)
        return frequency

    def _classify(self, sample):
        """
        Classification using Bayes rule p(y|x) = p(x|y) * p(y) / p(x)
        p(y|x) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        p(x|y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _calculate_likelihood)
        p(y)   - Prior (given by _calculate_prior)
        p(x)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.

         Classifies the sample as the class that results in the largest P(Y|X) (posterior)

        :param sample: ndarray of shape (n_features)
            Sample to be classified

        :return: int
            Class with largest posterior for the given sample
        """
        posteriors = []
        # Go through list of classes
        for i, c in enumerate(self.classes):
            # Initialize posterior as prior
            posterior = self._calculate_prior(c)
            # Naive assumption (independence):
            # p(x1, x2, x3|y) = p(x1|y) * p(x2|y) * p(x3|y)
            # Posterior is product of prior and likelihoods (ignoring scaling factor)
            for feature_value, param in zip(sample, self.parameters[i]):
                # Likelihood of feature value given distribution of feature values given y
                likelihood = self._calculate_likelihood(param['mean'], param['var'], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        # Return the class with the largest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """

        :param X: ndarray of shape (n_samples, n_features)
            Test data

        :return: ndarray of shape (n_samples, )
            Predictions for test data
        """
        y_pred = [self._classify(sample) for sample in X]
        return y_pred
