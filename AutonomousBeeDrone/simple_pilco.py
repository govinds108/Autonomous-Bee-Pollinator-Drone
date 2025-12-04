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
        state = start_state.copy()
        cost = 0
        for _ in range(self.horizon):
            action = self.controller.compute_action(state)
            delta = self.gp_predict(state, action)
            state = state + delta

            # State layout: [x_err, area_norm]
            # x_err: centering error in [-1, 1], where 0 = centered
            # area_norm: normalized bounding box area in [0, 1]
            
            x_err = float(state[0]) if state.shape[0] > 0 else 0.0
            area_norm = float(np.clip(state[1], 0.0, 1.0)) if state.shape[0] > 1 else 0.0
            
            # Exponential area reward: incentivizes getting closer (larger area)
            area_cost = np.exp(1.0) - np.exp(area_norm)
            
            # Quadratic centering penalty: incentivizes keeping flower centered
            centering_cost = x_err * x_err  # Ranges from 0 (centered) to 1 (at edges)
            
            # Combined cost: 70% area, 30% centering
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
