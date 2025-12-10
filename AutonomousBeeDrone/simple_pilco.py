import numpy as np
from simple_gp import SimpleGP

class SimplePILCO:
    def __init__(self, X, U, Y, controller, horizon=30, lr=0.01):
        self.X = X
        self.U = U
        self.Y = Y
        self.controller = controller
        self.horizon = horizon
        self.lr = lr

        XA = np.hstack((X, U))
        self.gps = [SimpleGP(XA, Y[:, i:i+1]) for i in range(Y.shape[1])]

    def gp_predict(self, state, action):
        xa = np.hstack((state, action)).reshape(1, -1)
        return np.array([gp.predict(xa)[0].item() for gp in self.gps])

    def rollout(self, start_state):
        """Rollout: predict trajectory and accumulate cost.
        
        Supports multi-dimensional actions from controller.
        
        Args:
            start_state: initial state (typically 2D: [x_err, area_norm])
        Returns:
            float: accumulated cost over horizon
        """
        state = start_state.copy()
        cost = 0
        for _ in range(self.horizon):
            # Get action from controller (can be multi-dimensional)
            action = self.controller.compute_action(state)
            # Ensure action is properly shaped for GP prediction
            action = np.atleast_1d(action).flatten()
            
            # Predict state delta using GPs
            delta = self.gp_predict(state, action)
            state = state + delta

            # State layout: [x_err, area_norm]
            # x_err: centering error in [-1, 1], where 0 = centered
            # area_norm: normalized bounding box area in [0, 1]
            
            x_err = float(state[0]) if state.shape[0] > 0 else 0.0
            area_norm = float(np.clip(state[1], 0.0, 1.0)) if state.shape[0] > 1 else 0.0
            
            # Exponential area reward: incentivizes getting closer (larger area)
            # As drone approaches, area_norm increases, making this term negative (reward)
            area_cost = np.exp(1.0) - np.exp(area_norm)
            
            # Quadratic centering penalty: incentivizes keeping flower centered
            # Penalty is 0 when centered (x_err=0), grows as drone drifts left/right
            centering_cost = x_err * x_err  # Ranges from 0 (centered) to 1 (at edges)
            
            # Combined cost: 70% area reward, 30% centering penalty
            # This biases the controller to prioritize approaching while maintaining center
            cost += 0.7 * area_cost + 0.3 * centering_cost
        return cost

    def optimize(self, start_state, iters=20):
        for _ in range(iters):
            grad = np.zeros_like(self.controller.weights)
            base_cost = self.rollout(start_state)
            eps = 1e-4

            # Finite-difference gradient over flattened weight vector
            for i in range(grad.size):
                orig = float(self.controller.weights[i])

                self.controller.weights[i] = orig + eps
                c_plus = self.rollout(start_state)

                self.controller.weights[i] = orig - eps
                c_minus = self.rollout(start_state)

                grad[i] = (c_plus - c_minus) / (2 * eps)
                self.controller.weights[i] = orig

            self.controller.weights = self.controller.weights - (self.lr * grad)

        return self.controller
