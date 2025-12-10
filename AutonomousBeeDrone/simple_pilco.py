import numpy as np
from simple_gp import SimpleGP

class SimplePILCO:
    def __init__(self, X, U, Y, controller, horizon=30, lr=0.01):
        """
        X: states   (N, state_dim)
        U: actions  (N, 2) → [yaw, forward]
        Y: deltas   (N, state_dim)
        """
        self.X = X
        self.U = U
        self.Y = Y
        self.controller = controller
        self.horizon = horizon
        self.lr = lr

        # Build GP dynamics models
        XA = np.hstack((X, U))  # (state + action)
        self.gps = [SimpleGP(XA, Y[:, i:i+1]) for i in range(Y.shape[1])]

    def gp_predict(self, state, action):
        xa = np.hstack((state, action)).reshape(1, -1)
        return np.array([gp.predict(xa)[0].item() for gp in self.gps])

    def rollout(self, start_state):
        """
        state = [x_err, area_norm]
        action = [yaw, forward]
        cost = MINIMIZED (lower = better)
        """
        state = start_state.copy()
        total_cost = 0.0

        for _ in range(self.horizon):
            action = self.controller.compute_action(state)  # 2-dim action

            # Predict next state using GP model
            delta = self.gp_predict(state, action)
            state = state + delta

            # Extract states safely
            x_err = float(state[0])
            area_norm = float(np.clip(state[1], 0.0, 1.0))

            yaw = float(action[0])
            forward = float(action[1])

            # ----------------------
            # Reward → expressed as cost for PILCO
            # ----------------------

            # 1. Keep flower centered
            centering_cost = x_err * x_err

            # 2. Encourage approaching flower
            approach_cost = np.exp(1.0 - area_norm)

            # 3. Encourage forward thrust
            forward_cost = np.exp(-forward)

            # Weighted total
            step_cost = (
                0.3 * centering_cost +
                0.4 * approach_cost +
                0.3 * forward_cost
            )

            total_cost += step_cost

        return total_cost

    def optimize(self, start_state, iters=20):
        for _ in range(iters):
            base_cost = self.rollout(start_state)
            grad = np.zeros_like(self.controller.weights)
            eps = 1e-4

            for i in range(grad.size):
                orig = float(self.controller.weights[i])

                # +eps
                self.controller.weights[i] = orig + eps
                c_plus = self.rollout(start_state)

                # -eps
                self.controller.weights[i] = orig - eps
                c_minus = self.rollout(start_state)

                grad[i] = (c_plus - c_minus) / (2 * eps)
                self.controller.weights[i] = orig

            # Gradient descent
            self.controller.weights -= self.lr * grad

        return self.controller
