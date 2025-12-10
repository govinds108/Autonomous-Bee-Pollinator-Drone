import numpy as np

class PILCOControllerWrapper:
    """
    Wraps RBFController so output matches drone action format: [yaw, forward]
    """

    def __init__(self, controller):
        self.controller = controller

    def compute_action(self, state):
        # Ensure vector format
        state = np.array(state).flatten()
        action = self.controller.compute_action(state)

        # Clip both outputs to [-1, 1]
        return np.clip(action, -1.0, 1.0)
