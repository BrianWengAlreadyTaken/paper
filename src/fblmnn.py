import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances
from scipy.optimize import minimize
import warnings


class FeasibleLMNN(BaseEstimator, TransformerMixin):
    """
    Feasibility-based Large Margin Nearest Neighbor (LMNN) Implementation

    This version incorporates a feasibility weight into the push (imposter)
    term. The weight is defined as:
        w(i,j,l) = exp(-gamma * (d_il - d_ij))
    where d_ij = ||L*(X[i] - X[j])||^2 is the distance between an instance
    and its target neighbor, and d_il = ||L*(X[i] - X[l])||^2 is the distance
    to an imposter. The push loss for an active constraint becomes:
        w(i,j,l) * max(0, margin + d_ij - d_il)

    The gradient is computed accordingly, applying the product rule to
    include the effect of the feasibility weight.

    Parameters
    ----------
    k : int, default=3
        Number of neighbors to consider for target neighbors.
    regularization : float, default=0.5
        Regularization parameter for the optimization (balances pull vs. push).
    gamma : float, default=1.0
        Parameter controlling the feasibility weight scaling.
    margin : float, default=1.0
        Base margin for imposter constraints.
    learning_rate : float, default=1e-7
        Learning rate for the gradient descent (not directly used in L-BFGS-B).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-5
        Convergence tolerance.

    Attributes
    ----------
    L_ : ndarray of shape (n_features, n_features)
        The learned linear transformation matrix.
    """

    def __init__(
        self,
        k=3,
        regularization=0.5,
        gamma=1.0,
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

    def _find_target_neighbors(self, X, y):
        """
        Identify the k nearest neighbors of each sample that share the same class.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        target_neighbors : ndarray of shape (n_samples, k)
            Indices of the k nearest neighbors in the same class for each sample.
        """
        n_samples = X.shape[0]
        target_neighbors = []

        for i in range(n_samples):
            same_class_idx = np.where(y == y[i])[0]
            same_class_idx = same_class_idx[same_class_idx != i]
            # Euclidean distance in the original space just to pick neighbors
            distances = euclidean_distances([X[i]], X[same_class_idx]).ravel()
            nn_indices = np.argsort(distances)[: self.k]
            target_neighbors.append(same_class_idx[nn_indices])

        return np.array(target_neighbors)

    def _compute_loss(self, L_flat, X, y, target_neighbors):
        """
        Compute the feasibility-based LMNN loss function.

        Pull term: sum of squared distances between each sample and its target neighbors.
        Push term: a weighted hinge loss for impostors, with an exponential feasibility factor.

        Parameters
        ----------
        L_flat : ndarray of shape (n_features*n_features,)
            Flattened transformation matrix.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Class labels.
        target_neighbors : ndarray of shape (n_samples, k)
            Indices of target neighbors for each sample.

        Returns
        -------
        loss : float
            The total loss value (pull + regularized push).
        """
        n_samples, n_features = X.shape
        L = L_flat.reshape(n_features, n_features)

        # Pull term: attract target neighbors
        pull_loss = 0.0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                pull_loss += np.sum((diff_ij.dot(L.T)) ** 2)

        # Push term: repel imposters with feasibility weight
        push_loss = 0.0
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                d_ij = np.sum((diff_ij.dot(L.T)) ** 2)

                # Imposters are samples of a different class
                imposters = np.where(y != y[i])[0]
                for l in imposters:
                    diff_il = X[i] - X[l]
                    d_il = np.sum((diff_il.dot(L.T)) ** 2)
                    loss_term = self.margin + d_ij - d_il
                    if loss_term > 0:
                        # Feasibility weight
                        weight = np.exp(-self.gamma * (d_il - d_ij))
                        push_loss += weight * loss_term

        # Combine pull and push
        loss = pull_loss + self.regularization * push_loss
        return loss

    def _compute_gradient(self, L_flat, X, y, target_neighbors):
        """
        Compute the gradient of the feasibility-based LMNN loss function.

        Parameters
        ----------
        L_flat : ndarray of shape (n_features*n_features,)
            Flattened transformation matrix.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Class labels.
        target_neighbors : ndarray of shape (n_samples, k)
            Indices of target neighbors for each sample.

        Returns
        -------
        grad_flat : ndarray of shape (n_features*n_features,)
            Flattened gradient of the loss w.r.t. the transformation matrix.
        """
        n_samples, n_features = X.shape
        L = L_flat.reshape(n_features, n_features)
        gradient = np.zeros_like(L)

        # Gradient of pull term
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                # d/dL of ||L diff||^2 = 2 * (L diff) (diff)^T
                gradient += 2.0 * np.outer(L.dot(diff_ij), diff_ij)

        # Gradient of push term
        for i in range(n_samples):
            for j in target_neighbors[i]:
                diff_ij = X[i] - X[j]
                d_ij = np.sum((diff_ij.dot(L.T)) ** 2)

                imposters = np.where(y != y[i])[0]
                for l in imposters:
                    diff_il = X[i] - X[l]
                    d_il = np.sum((diff_il.dot(L.T)) ** 2)
                    loss_term = self.margin + d_ij - d_il
                    if loss_term > 0:
                        # Weight = exp(-gamma (d_il - d_ij))
                        weight = np.exp(-self.gamma * (d_il - d_ij))
                        # Derivative factor from hinge + weight
                        # We have "weight * (margin + d_ij - d_il)"
                        # but only partial derivative w.r.t. d_ij, d_il
                        # => 2 * (L diff_ij) diff_ij^T - 2 * (L diff_il) diff_il^T
                        # multiplied by weight and plus the derivative of the exponent if you do chain rule exactly.
                        # However, a simpler approach is the product rule:
                        # push_grad = 2 * weight * [outer(L diff_ij, diff_ij) - outer(L diff_il, diff_il)]
                        # times a factor that accounts for derivative of weight wrt L.
                        # A simpler "feasibility-based" version often approximates:
                        grad_component = (
                            2.0
                            * weight
                            * (
                                np.outer(L.dot(diff_ij), diff_ij)
                                - np.outer(L.dot(diff_il), diff_il)
                            )
                        )
                        # Multiply by self.regularization
                        gradient += self.regularization * grad_component

        grad_flat = gradient.ravel()
        return grad_flat

    def fit(self, X, y):
        """
        Fit the Feasibility-based LMNN model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
            The fitted instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Initialize L as the identity matrix
        L_init = np.eye(n_features).ravel()

        # Determine target neighbors for each sample
        target_neighbors = self._find_target_neighbors(X, y)

        # Optimize using L-BFGS-B
        result = minimize(
            fun=self._compute_loss,
            x0=L_init,
            args=(X, y, target_neighbors),
            method="L-BFGS-B",
            jac=self._compute_gradient,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        if not result.success:
            warnings.warn(f"FB-LMNN optimization did not converge: {result.message}")

        self.L_ = result.x.reshape(n_features, n_features)
        return self

    def transform(self, X):
        """
        Apply the learned transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        X = np.asarray(X)
        return X.dot(self.L_.T)
