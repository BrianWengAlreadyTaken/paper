import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
import warnings


class LMNN(BaseEstimator, TransformerMixin):
    """
    Large Margin Nearest Neighbor (LMNN) Implementation with Multi-Pass Support

    Parameters:
    -----------
    k : int, default=3
        Number of neighbors to consider for target neighbors
    regularization : float, default=0.5
        Regularization parameter for the optimization
    learning_rate : float, default=1e-7
        Learning rate for the gradient descent (used in initialization)
    max_iter : int, default=1000
        Maximum number of iterations per pass
    tol : float, default=1e-5
        Convergence tolerance for optimization
    passes : int, default=1
        Number of passes over the data for optimization

    Attributes:
    -----------
    L_ : array-like, shape (n_features, n_features)
        The learned linear transformation matrix
    """

    def __init__(
        self,
        k=3,
        regularization=0.5,
        learning_rate=1e-7,
        max_iter=1000,
        tol=1e-5,
        passes=1,
    ):
        self.k = k
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.passes = passes

    def _find_target_neighbors(self, X, y, L=None):
        """Find target neighbors (same class) for each instance using current metric."""
        n_samples = X.shape[0]
        target_neighbors = []

        # Transform X with current L if provided (for multi-pass updates)
        if L is not None:
            X_transformed = X.dot(L.T)
        else:
            X_transformed = X

        for i in range(n_samples):
            same_class = np.where(y == y[i])[0]
            same_class = same_class[same_class != i]
            distances = euclidean_distances(
                [X_transformed[i]], X_transformed[same_class]
            ).ravel()
            target_neighbors.append(same_class[np.argsort(distances)[: self.k]])

        return np.array(target_neighbors)

    def _compute_loss(self, L_flat, X, y, target_neighbors):
        """Compute the LMNN loss function."""
        n_samples, n_features = X.shape
        L = L_flat.reshape(n_features, n_features)

        pull_loss = 0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff = X[i] - X[j]
                pull_loss += np.sum((diff.dot(L.T)) ** 2)

        push_loss = 0
        margin = 1.0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                dist_ij = np.sum((diff_ij.dot(L.T)) ** 2)

                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = X[i] - X[l]
                    dist_il = np.sum((diff_il.dot(L.T)) ** 2)

                    push_loss += max(0, margin + dist_ij - dist_il)

        return pull_loss + self.regularization * push_loss

    def _compute_gradient(self, L_flat, X, y, target_neighbors):
        """Compute the gradient of the LMNN loss function."""
        n_samples, n_features = X.shape
        L = L_flat.reshape(n_features, n_features)
        gradient = np.zeros_like(L)

        # Gradient of pull term
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff = X[i] - X[j]
                gradient += 2 * np.outer(L.dot(diff), diff)

        # Gradient of push term
        margin = 1.0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                dist_ij = np.sum((diff_ij.dot(L.T)) ** 2)

                different_class = y != y[i]
                for l in np.where(different_class)[0]:
                    diff_il = X[i] - X[l]
                    dist_il = np.sum((diff_il.dot(L.T)) ** 2)

                    if margin + dist_ij - dist_il > 0:  # Active constraint
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
        """
        Fit the LMNN model with multiple passes

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Returns self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize L as identity matrix for the first pass
        L_current = np.eye(n_features)

        for pass_idx in range(self.passes):
            target_neighbors = self._find_target_neighbors(X, y, L=L_current)

            result = minimize(
                fun=self._compute_loss,
                x0=L_current.ravel(),
                args=(X, y, target_neighbors),
                method="L-BFGS-B",
                jac=self._compute_gradient,
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )

            if not result.success:
                warnings.warn(
                    f"LMNN optimization did not converge in pass {pass_idx + 1}: {result.message}"
                )

            L_current = result.x.reshape(n_features, n_features)

        self.L_ = L_current
        return self

    def transform(self, X):
        """
        Apply the learned transformation

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform

        Returns:
        --------
        X_new : array, shape (n_samples, n_features)
            Transformed data
        """
        X = np.asarray(X)
        return X.dot(self.L_.T)
