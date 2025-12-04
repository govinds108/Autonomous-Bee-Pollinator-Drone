import numpy as np

class RBFController:
    def __init__(self, state_dim, num_basis=10, action_dim=1):
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.action_dim = action_dim
        self.centers = np.random.randn(num_basis, state_dim)
        self.sigma = 0.5
        # Store weights as a flat vector of length num_basis * action_dim to
        # make numerical optimization (finite-diff) simple in SimplePILCO.
        self.weights = np.random.randn(num_basis * action_dim) * 0.1

    def _basis(self, state):
        diff = state.reshape(1, -1) - self.centers
        return np.exp(-np.sum(diff**2, axis=1) / (2 * self.sigma**2))

    def compute_action(self, state):
        phi = self._basis(state)  # shape: (num_basis,)
        W = self.weights.reshape(self.num_basis, self.action_dim)
        action = phi @ W  # shape: (action_dim,)
        return action.flatten()
