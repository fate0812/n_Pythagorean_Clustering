import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist


class nPyFCM:
    def __init__(self, n_clusters=4, max_iter=100, m=2, error=1e-6,n_pyth = 2,alpha = 1):
        super().__init__()
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.n_pyth = n_pyth
        self.alpha = alpha

    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters
        centers = []

        u = np.random.dirichlet(np.ones(C), size=N)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return centers

    def next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).transpose()

    def next_u(self, X, centers):
        return self._predict(X, centers)

    def _predict(self, X, centers):
        power = self.n_pyth / self.alpha
        power1 = float(2 / (self.m - 1))
        temp = cdist(X, centers) ** power1
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        membership  = 1 / denominator_.sum(2)
        return (1 - (1 - membership ** self.alpha) ** power) ** (1 / self.n_pyth)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X, self.centers)
        return u.T
