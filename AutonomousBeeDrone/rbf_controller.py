import numpy as np

class RBFController:
    """RBF controller that maps state to actions.
    
    Supports multi-dimensional outputs:
    - weights stored as flat vector for efficient optimization
    - compute_action outputs action_dim-dimensional action
    """
    def __init__(self, state_dim, num_basis=10, action_dim=2):
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.action_dim = action_dim  # Default to 2D: [yaw, forward]
        self.centers = np.random.randn(num_basis, state_dim)
        self.sigma = 0.5
        # Store weights as a flat vector of length num_basis * action_dim to
        # make numerical optimization (finite-diff) simple in SimplePILCO.
        # Initialization: slightly smaller to prevent extreme actions
        self.weights = np.random.randn(num_basis * action_dim) * 0.05

    def _basis(self, state):
        """Compute RBF basis functions.
        
        Args:
            state: numpy array of shape (state_dim,)
        Returns:
            numpy array of shape (num_basis,) with basis activations
        """
        diff = state.reshape(1, -1) - self.centers
        return np.exp(-np.sum(diff**2, axis=1) / (2 * self.sigma**2))

    def compute_action(self, state):
        """Compute action from state using RBF basis functions.
        
        Args:
            state: numpy array of shape (state_dim,)
        Returns:
            numpy array of shape (action_dim,) with action values
        """
        phi = self._basis(state)  # shape: (num_basis,)
        W = self.weights.reshape(self.num_basis, self.action_dim)
        action = phi @ W  # shape: (action_dim,)
        return action.flatten()
