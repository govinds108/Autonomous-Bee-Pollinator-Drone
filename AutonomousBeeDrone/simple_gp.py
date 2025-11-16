import numpy as np

class SimpleGP:
    def __init__(self, X, Y, lengthscale=1.0, variance=1.0, noise=1e-4):
        self.X = np.array(X, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self._compute_kernel()

    def rbf_kernel(self, A, B):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        diff = A[:, None, :] - B[None, :, :]
        sqdist = np.sum(diff**2, axis=2)
        return self.variance * np.exp(-0.5 * sqdist / (self.lengthscale**2))

    def _compute_kernel(self):
        self.K = self.rbf_kernel(self.X, self.X)
        self.K += np.eye(len(self.K)) * self.noise
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_star):
        X_star = np.atleast_2d(X_star)
        K_star = self.rbf_kernel(X_star, self.X)
        mean = K_star @ self.K_inv @ self.Y
        var = self.rbf_kernel(X_star, X_star) - K_star @ self.K_inv @ K_star.T
        var = np.maximum(np.diag(var), 0)
        return mean, var
