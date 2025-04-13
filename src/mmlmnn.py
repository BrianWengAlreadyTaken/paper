import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import warnings


class MultiMetricLMNN(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, n_clusters=3, regularization=0.5, max_iter=1000, tol=1e-5):
        self.k = k
        self.n_clusters = n_clusters
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol

    def _find_target_neighbors(self, X, y, L=None):
        n_samples = X.shape[0]
        target_neighbors = []
        valid_indices = []
        X_transformed = X.dot(L.T) if L is not None else X

        for i in range(n_samples):
            same_class = np.where(y == y[i])[0]
            same_class = same_class[same_class != i]
            if len(same_class) < self.k:
                continue
            distances = euclidean_distances(
                [X_transformed[i]], X_transformed[same_class]
            ).ravel()
            target_neighbors.append(same_class[np.argsort(distances)[: self.k]])
            valid_indices.append(i)

        if not target_neighbors:
            return np.empty((0, self.k), dtype=int), np.empty(0, dtype=int)
        return np.array(target_neighbors), np.array(valid_indices)

    def _compute_loss(self, L_flat, X, y, target_neighbors, valid_indices):
        n_features = X.shape[1]
        L = L_flat.reshape(n_features, n_features)

        pull_loss = 0
        for idx, i in enumerate(valid_indices):
            for j in target_neighbors[idx]:
                diff = X[i] - X[j]
                pull_loss += np.sum((diff.dot(L.T)) ** 2)

        push_loss = 0
        margin = 1.0
        for idx, i in enumerate(valid_indices):
            for j in target_neighbors[idx]:
                diff_ij = X[i] - X[j]
                dist_ij = np.sum((diff_ij.dot(L.T)) ** 2)
                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = X[i] - X[l]
                    dist_il = np.sum((diff_il.dot(L.T)) ** 2)
                    push_loss += max(0, margin + dist_ij - dist_il)

        return pull_loss + self.regularization * push_loss

    def _compute_gradient(self, L_flat, X, y, target_neighbors, valid_indices):
        n_features = X.shape[1]
        L = L_flat.reshape(n_features, n_features)
        gradient = np.zeros_like(L)

        for idx, i in enumerate(valid_indices):
            for j in target_neighbors[idx]:
                diff = X[i] - X[j]
                gradient += 2 * np.outer(L.dot(diff), diff)

        margin = 1.0
        for idx, i in enumerate(valid_indices):
            for j in target_neighbors[idx]:
                diff_ij = X[i] - X[j]
                dist_ij = np.sum((diff_ij.dot(L.T)) ** 2)
                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = X[i] - X[l]
                    dist_il = np.sum((diff_il.dot(L.T)) ** 2)
                    if margin + dist_ij - dist_il > 0:
                        gradient += (
                            2
                            * self.regularization
                            * (
                                np.outer(L.dot(diff_ij), diff_ij)
                                - np.outer(L.dot(diff_il), diff_il)
                            )
                        )

        return gradient.ravel()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        self.n_clusters = min(self.n_clusters, n_samples)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit(X)  # pyright: ignore
        self.cluster_labels_ = kmeans.labels_
        self.kmeans_ = kmeans
        self.L_matrices_ = []

        for cluster in range(self.n_clusters):
            cluster_idx = np.where(self.cluster_labels_ == cluster)[0]
            if len(cluster_idx) <= self.k:
                self.L_matrices_.append(np.eye(n_features))
                continue

            X_cluster = X[cluster_idx]
            y_cluster = y[cluster_idx]
            target_neighbors, valid_indices = self._find_target_neighbors(
                X_cluster, y_cluster
            )
            if target_neighbors.size == 0:
                self.L_matrices_.append(np.eye(n_features))
                continue

            L_init = np.eye(n_features).ravel()
            result = minimize(
                fun=self._compute_loss,
                x0=L_init,
                args=(X_cluster, y_cluster, target_neighbors, valid_indices),
                method="L-BFGS-B",
                jac=self._compute_gradient,
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )

            if not result.success:
                warnings.warn(f"Optimization failed for cluster {cluster}.")
                self.L_matrices_.append(np.eye(n_features))
            else:
                self.L_matrices_.append(result.x.reshape(n_features, n_features))

        return self

    def transform(self, X):
        X = np.asarray(X)
        X_transformed = np.zeros_like(X)
        cluster_labels = self.kmeans_.predict(X)

        for cluster in range(self.n_clusters):
            cluster_idx = np.where(cluster_labels == cluster)[0]
            if cluster_idx.size > 0:
                X_transformed[cluster_idx] = X[cluster_idx].dot(
                    self.L_matrices_[cluster].T
                )

        return X_transformed
