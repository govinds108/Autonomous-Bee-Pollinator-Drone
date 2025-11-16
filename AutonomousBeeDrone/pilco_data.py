from collections import deque
import numpy as np

class ReplayBufferPILCO:
    def __init__(self, max_size=50000):
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)

    def add(self, s, a, s_next):
        self.states.append(np.array(s, dtype=np.float32))
        self.actions.append(np.array(a, dtype=np.float32))
        self.next_states.append(np.array(s_next, dtype=np.float32))

    def get(self):
        S = np.array(self.states)
        A = np.array(self.actions)
        S2 = np.array(self.next_states)
        return S, A, S2
