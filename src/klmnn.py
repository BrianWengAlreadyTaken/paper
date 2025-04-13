import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize
import warnings


class KernelLMNN(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, regularization=0.5, gamma=1.0, max_iter=1000, tol=1e-5):
        self.k = k
        self.regularization = regularization
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def _compute_kernel(self, X, Y=None):
        return rbf_kernel(X, Y, gamma=self.gamma)

    def _find_target_neighbors(self, K, y):
        n_samples = K.shape[0]
        target_neighbors = []
        for i in range(n_samples):
            same_class = np.where(y == y[i])[0]
            same_class = same_class[same_class != i]
            distances = np.sqrt(
                np.maximum(
                    0, K[i, i] - 2 * K[i, same_class] + K[same_class, same_class]
                )
            )
            target_neighbors.append(same_class[np.argsort(distances)[: self.k]])
        return np.array(target_neighbors)

    def _compute_loss(self, M_flat, K, y, target_neighbors):
        n_samples = K.shape[0]
        M = M_flat.reshape(n_samples, n_samples)
        M = (M + M.T) / 2

        pull_loss = 0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff = K[i] - K[j]
                pull_loss += diff.dot(M).dot(diff)

        push_loss = 0
        margin = 1.0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = K[i] - K[j]
                dist_ij = diff_ij.dot(M).dot(diff_ij)
                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = K[i] - K[l]
                    dist_il = diff_il.dot(M).dot(diff_il)
                    push_loss += max(0, margin + dist_ij - dist_il)

        return pull_loss + self.regularization * push_loss

    def _compute_gradient(self, M_flat, K, y, target_neighbors):
        n_samples = K.shape[0]
        M = M_flat.reshape(n_samples, n_samples)
        gradient = np.zeros_like(M)

        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff = K[i] - K[j]
                gradient += np.outer(diff, diff)

        margin = 1.0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = K[i] - K[j]
                dist_ij = diff_ij.dot(M).dot(diff_ij)
                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = K[i] - K[l]
                    dist_il = diff_il.dot(M).dot(diff_il)
                    if margin + dist_ij - dist_il > 0:
                        gradient += self.regularization * (
                            np.outer(diff_ij, diff_ij) - np.outer(diff_il, diff_il)
                        )

        return gradient.ravel()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train_ = X
        n_samples = X.shape[0]

        K = self._compute_kernel(X)
        target_neighbors = self._find_target_neighbors(K, y)

        M_init = np.eye(n_samples).ravel()
        result = minimize(
            fun=self._compute_loss,
            x0=M_init,
            args=(K, y, target_neighbors),
            method="L-BFGS-B",
            jac=self._compute_gradient,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        if not result.success:
            warnings.warn("KernelLMNN optimization did not converge.")

        self.M_ = result.x.reshape(n_samples, n_samples)
        self.M_ = (self.M_ + self.M_.T) / 2
        return self

    def transform(self, X):
        X = np.asarray(X)
        K_test = self._compute_kernel(X, self.X_train_)
        return K_test.dot(self.M_)
