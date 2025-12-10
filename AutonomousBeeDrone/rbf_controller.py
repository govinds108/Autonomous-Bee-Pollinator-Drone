import numpy as np

class RBFController:
    def __init__(self, state_dim, num_basis=10, action_dim=2):
        """
        action_dim = 2 → [yaw, forward]
        """
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.action_dim = action_dim

        # RBF centers & width
        self.centers = np.random.randn(num_basis, state_dim)
        self.sigma = 0.5

        # Flattened weight vector
        self.weights = np.random.randn(num_basis * action_dim) * 0.1

    def _basis(self, state):
        diff = state.reshape(1, -1) - self.centers
        return np.exp(-np.sum(diff**2, axis=1) / (2 * self.sigma**2))

    def compute_action(self, state):
        phi = self._basis(state)  # (num_basis,)
        W = self.weights.reshape(self.num_basis, self.action_dim)

        # Raw actions
        action = phi @ W  # shape: (2,)

        # Squash to [-1, 1]
        return np.tanh(action)
