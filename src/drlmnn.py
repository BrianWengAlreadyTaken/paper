import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
import warnings


class DRLMNN(BaseEstimator, TransformerMixin):
    def __init__(
        self, k=3, n_components=2, regularization=0.5, max_iter=1000, tol=1e-5
    ):
        self.k = k
        self.n_components = n_components
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol

    def _find_target_neighbors(self, X, y, L=None):
        n_samples = X.shape[0]
        target_neighbors = []
        X_transformed = X.dot(L.T) if L is not None else X

        for i in range(n_samples):
            same_class = np.where(y == y[i])[0]
            same_class = same_class[same_class != i]
            distances = euclidean_distances(
                [X_transformed[i]], X_transformed[same_class]
            ).ravel()
            target_neighbors.append(same_class[np.argsort(distances)[: self.k]])

        return np.array(target_neighbors)

    def _compute_loss(self, L_flat, X, y, target_neighbors):
        n_features = X.shape[1]
        L = L_flat.reshape(self.n_components, n_features)

        pull_loss = 0
        for i in range(X.shape[0]):
            for j in target_neighbors[i]:
                diff = X[i] - X[j]
                pull_loss += np.sum((L.dot(diff)) ** 2)

        push_loss = 0
        margin = 1.0
        for i in range(X.shape[0]):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                dist_ij = np.sum((L.dot(diff_ij)) ** 2)
                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = X[i] - X[l]
                    dist_il = np.sum((L.dot(diff_il)) ** 2)
                    push_loss += max(0, margin + dist_ij - dist_il)

        return pull_loss + self.regularization * push_loss

    def _compute_gradient(self, L_flat, X, y, target_neighbors):
        n_features = X.shape[1]
        L = L_flat.reshape(self.n_components, n_features)
        gradient = np.zeros_like(L)

        for i in range(X.shape[0]):
            for j in target_neighbors[i]:
                diff = X[i] - X[j]
                gradient += 2 * np.outer(L.dot(diff), diff)

        margin = 1.0
        for i in range(X.shape[0]):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                dist_ij = np.sum((L.dot(diff_ij)) ** 2)
                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = X[i] - X[l]
                    dist_il = np.sum((L.dot(diff_il)) ** 2)
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

        target_neighbors = self._find_target_neighbors(X, y)

        L_init = np.random.randn(self.n_components, n_features).ravel()
        result = minimize(
            fun=self._compute_loss,
            x0=L_init,
            args=(X, y, target_neighbors),
            method="L-BFGS-B",
            jac=self._compute_gradient,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        if not result.success:
            warnings.warn("DR-LMNN optimization did not converge.")

        self.L_ = result.x.reshape(self.n_components, n_features)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X.dot(self.L_.T)
