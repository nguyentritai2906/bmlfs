from bmlfs.utils import euclidean_distance
import numpy as np


class KMeans():
    """A simple clustering method that forms k clusters by iteratively
    reassigning samples to the closest centroids and after that moves the
    centroids to the center of the newly-formed clusters.


    :k: int
        The number of clusters the algorithm will form.
    :max_iterations: int
        The number of iterations the algorithm will run for if it does
        not converge before that.
    """
    def __init__(self, k=2, max_iter=500):
        self.k = k
        self.max_iter = max_iter

    def _init_random_centroids(self, X):
        """ Initialize the centroids as k random samples of X

        :X: ndarray of shape (n_samples, n_features)
        :returns: ndarray of shape (n_clusters, n_features)

        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, sample, centroids):
        """ Return the index of the closest centroid to the sample

        :sample: ndarray of shape (1, n_features)
        :centroids: ndarray of shape (n_clusters, n_features)
        :returns: int

        """
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def _create_clusters(self, centroids, X):
        """ Assign the samples to the closest centroids to create clusters

        :centroids: ndarray of shape (n_clusters, n_features)
        :X: ndarray of shape (n_samples, n_features)
        :returns: list[list] of centroid indexes for each cluster

        """
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def _calculate_centroids(self, clusters, X):
        """ Calculate new centroids as the means of the samples in each cluster

        :clusters: list[list] of centroid indexes for each cluster
        :X: ndarray of shape (n_samples, n_features)
        :returns: ndarray of shape (n_clusters, n_features)

        """
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def _get_cluster_labels(self, clusters, X):
        """ Classify samples as the index of their clusters

        :clusters: list[list] of centroid indexes for each cluster
        :X: ndarray of shape (n_samples, n_features)
        :returns: ndarray of shape (n_samples,)

        """
        y_pred = np.zeros(X.shape[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        """ Do K-Means clustering and return cluster indices

        :X: ndarray of shape (n_samples, n_features)
        :returns: ndarray of shape (n_samples,)

        """
        centroids = self._init_random_centroids(X)

        for _ in range(self.max_iter):
            clusters = self._create_clusters(centroids, X)
            prev_centroids = centroids
            centroids = self._calculate_centroids(clusters, X)
            diff = centroids - prev_centroids
            if not diff.any():
                break

        return self._get_cluster_labels(clusters, X)
