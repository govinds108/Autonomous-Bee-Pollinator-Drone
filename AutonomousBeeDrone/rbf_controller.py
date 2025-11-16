import numpy as np

class RBFController:
    def __init__(self, state_dim, num_basis=10):
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.centers = np.random.randn(num_basis, state_dim)
        self.sigma = 0.5
        self.weights = np.random.randn(num_basis, 1) * 0.1

    def _basis(self, state):
        diff = state.reshape(1, -1) - self.centers
        return np.exp(-np.sum(diff**2, axis=1) / (2 * self.sigma**2))

    def compute_action(self, state):
        phi = self._basis(state)
        return (phi @ self.weights).flatten()
