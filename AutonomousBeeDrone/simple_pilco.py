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

            x_err = state[0]                   # yaw-only
            cost += x_err * x_err
        return cost

    def optimize(self, start_state, iters=20):
        for _ in range(iters):
            grad = np.zeros_like(self.controller.weights)
            base_cost = self.rollout(start_state)
            eps = 1e-4

            for i in range(len(grad)):
                orig = self.controller.weights[i].copy()

                self.controller.weights[i] = orig + eps
                c_plus = self.rollout(start_state)

                self.controller.weights[i] = orig - eps
                c_minus = self.rollout(start_state)

                grad[i] = (c_plus - c_minus) / (2 * eps)
                self.controller.weights[i] = orig

            self.controller.weights -= self.lr * grad

        return self.controller
