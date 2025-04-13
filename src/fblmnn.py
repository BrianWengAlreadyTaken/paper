import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings


class FeasibleLMNN(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        k=3,
        regularization=0.5,
        gamma=0.1,
        margin=1.0,
        learning_rate=1e-7,
        max_iter=1000,
        tol=1e-5,
    ):
        self.k = k
        self.regularization = regularization
        self.gamma = gamma
        self.margin = margin
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.scaler_ = StandardScaler()

    def _find_target_neighbors(self, X, y):
        n_samples = X.shape[0]
        target_neighbors = []

        for i in range(n_samples):
            same_class_idx = np.where(y == y[i])[0]
            same_class_idx = same_class_idx[same_class_idx != i]
            if len(same_class_idx) < self.k:
                continue
            distances = euclidean_distances([X[i]], X[same_class_idx]).ravel()
            nn_indices = np.argsort(distances)[: self.k]
            target_neighbors.append(same_class_idx[nn_indices])

        return (
            np.array(target_neighbors)
            if target_neighbors
            else np.empty((0, self.k), dtype=int)
        )

    def _compute_loss(self, L_flat, X, y, target_neighbors):
        n_samples, n_features = X.shape
        L = L_flat.reshape(n_features, n_features)

        pull_loss = 0.0
        for i in range(min(n_samples, len(target_neighbors))):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                pull_loss += np.sum((diff_ij.dot(L.T)) ** 2)

        push_loss = 0.0
        for i in range(min(n_samples, len(target_neighbors))):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                d_ij = np.sum((diff_ij.dot(L.T)) ** 2)
                imposters = np.where(y != y[i])[0]
                for l in imposters:
                    diff_il = X[i] - X[l]
                    d_il = np.sum((diff_il.dot(L.T)) ** 2)
                    loss_term = self.margin + d_ij - d_il
                    if loss_term > 0:
                        weight = np.exp(-self.gamma * (d_il - d_ij))
                        weight = np.clip(weight, 1e-10, 1e10)
                        push_loss += weight * loss_term

        return pull_loss + self.regularization * push_loss

    def _compute_gradient(self, L_flat, X, y, target_neighbors):
        n_samples, n_features = X.shape
        L = L_flat.reshape(n_features, n_features)
        gradient = np.zeros_like(L)

        for i in range(min(n_samples, len(target_neighbors))):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                gradient += 2.0 * np.outer(L.dot(diff_ij), diff_ij)

        for i in range(min(n_samples, len(target_neighbors))):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                d_ij = np.sum((diff_ij.dot(L.T)) ** 2)
                imposters = np.where(y != y[i])[0]
                for l in imposters:
                    diff_il = X[i] - X[l]
                    d_il = np.sum((diff_il.dot(L.T)) ** 2)
                    loss_term = self.margin + d_ij - d_il
                    if loss_term > 0:
                        weight = np.exp(-self.gamma * (d_il - d_ij))
                        weight = np.clip(weight, 1e-10, 1e10)
                        grad_hinge = 2.0 * (
                            np.outer(L.dot(diff_ij), diff_ij)
                            - np.outer(L.dot(diff_il), diff_il)
                        )
                        grad_weight = (
                            weight
                            * self.gamma
                            * 2.0
                            * (
                                np.outer(L.dot(diff_ij), diff_ij)
                                - np.outer(L.dot(diff_il), diff_il)
                            )
                        )
                        grad_component = weight * grad_hinge + loss_term * grad_weight
                        gradient += self.regularization * grad_component

        return gradient.ravel()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X = self.scaler_.fit_transform(X)
        n_samples, n_features = X.shape

        L_init = np.eye(n_features).ravel()
        target_neighbors = self._find_target_neighbors(X, y)

        result = minimize(
            fun=self._compute_loss,
            x0=L_init,
            args=(X, y, target_neighbors),
            method="L-BFGS-B",
            jac=self._compute_gradient,
            options={"maxiter": self.max_iter, "ftol": self.tol, "disp": False},
        )

        if not result.success:
            warnings.warn(f"FB-LMNN optimization did not converge: {result.message}")

        self.L_ = result.x.reshape(n_features, n_features)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X = self.scaler_.transform(X)
        return X.dot(self.L_.T)  # pyright: ignore
