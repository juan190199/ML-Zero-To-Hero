from __future__ import print_function, division
import numpy as np
from utils.data_manipulation import normalize
from utils.data_operation import euclidean_distance


class KMeans():
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def _init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, sample, centroids):
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def _create_clusters(self, centroids, X):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def _calculate_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def _get_cluster_labels(self, clusters, X):
        y_pred = np.empty(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        normalized_X = normalize(X)
        centroids = self._init_random_centroids(normalized_X)
        for _ in range(self.max_iterations):
            clusters = self._create_clusters(centroids, normalized_X)
            new_centroids = self._calculate_centroids(clusters, normalized_X)
            if np.array_equal(new_centroids, centroids):
                break
            centroids = new_centroids
        labels = self._get_cluster_labels(clusters, X)
        return labels