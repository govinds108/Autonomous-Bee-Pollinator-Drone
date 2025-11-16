import numpy as np

class PILCOControllerWrapper:
    def __init__(self, controller):
        self.controller = controller

    def compute_action(self, state):
        action = self.controller.compute_action(np.array(state))
        return np.clip(action, -1.0, 1.0)
