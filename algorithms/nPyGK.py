import numpy as np
from scipy.linalg import norm


class nPyGK:
    def __init__(self, n_clusters=4, max_iter=100, m=2, error=1e-6,n_pyth = 2,alpha = 1.5):
        super().__init__()
        self.u, self.centers, self.f = None, None, None
        self.clusters_count = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.n_pyth = n_pyth
        self.alpha = alpha

    def fit(self, z):
        N = z.shape[0]
        C = self.clusters_count
        centers = []

        u = np.random.dirichlet(np.ones(N), size=C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(z, u)
            f = self._covariance(z, centers, u)
            dist = self._distance(z, centers, f)
            u = self.next_u(dist)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.f = f
        self.u = u
        self.centers = centers
        return centers

    def next_centers(self, z, u):
        um = u ** self.m
        return ((um @ z).T / um.sum(axis=1)).T

    def _covariance(self, z, v, u):
        um = u ** self.m

        denominator = um.sum(axis=1).reshape(-1, 1, 1)
        temp = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        temp = np.matmul(temp, temp.transpose((0, 1, 3, 2)))
        numerator = um.transpose().reshape(um.shape[1], um.shape[0], 1, 1) * temp
        numerator = numerator.sum(0)

        return numerator / denominator

    
    def _distance(self, z, v, f):
        """Compute Mahalanobis distances."""
        diffs = np.expand_dims(z[:, None, :] - v[None, :, :], axis=3)
        regularized_cov = f + np.eye(f.shape[-1]) * 1e-6  # Regularization

        try:
            # Compute determinants with regularized covariance
            determinants = np.linalg.det(regularized_cov) ** (1 / self.m)

            # Compute the pseudo-inverse of covariance matrices
            inv_covariances = determinants[:, None, None] * np.linalg.pinv(regularized_cov)

            # Calculate Mahalanobis distances
            distances = np.matmul(
                np.matmul(diffs.transpose(0, 1, 3, 2), inv_covariances), diffs
            ).squeeze().T

            # Ensure no negative distances
            return np.fmax(distances, 1e-8)

        except np.linalg.LinAlgError as e:
            print(f"Error in SVD or determinant computation: {e}")
            raise
    
    def next_u(self, d):
        power1 = self.n_pyth / self.alpha
        power = float(1 / (self.m - 1))
        d = d.transpose()
        denominator_ = d.reshape((d.shape[0], 1, -1)).repeat(d.shape[-1], axis=1)
        denominator_ = np.power(d[:, None, :] / denominator_.transpose((0, 2, 1)), power)
        denominator_ = 1 / denominator_.sum(1)
        denominator_ = denominator_.transpose()
        return (1 - (1 - denominator_ ** self.alpha) ** power1) ** (1 / self.n_pyth)
    
    def predict(self, z):
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=0)

        dist = self._distance(z, self.centers, self.f)
        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=0)

        u = self.next_u(dist)
        return u
