import numpy as np

class PILCOControllerWrapper:
    """Wraps RBFController to return [yaw, forward]."""

    def __init__(self, controller):
        self.controller = controller

    def compute_action(self, state):
        state = np.array(state).flatten()
        action = self.controller.compute_action(state)
        return np.clip(action, -1.0, 1.0)
