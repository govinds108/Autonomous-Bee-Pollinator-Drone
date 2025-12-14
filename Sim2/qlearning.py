from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QLearnConfig:
    alpha: float = 0.2
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_decay: float = 0.97
    eps_min: float = 0.1


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: Optional[QLearnConfig] = None):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.cfg = cfg or QLearnConfig()
        self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.eps = float(self.cfg.eps_start)

    def act(self, state: int) -> int:
        if np.random.rand() < self.eps:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, s: int, a: int, r: float, s2: int) -> None:
        td_target = float(r) + self.cfg.gamma * float(np.max(self.q[s2]))
        self.q[s, a] = (1.0 - self.cfg.alpha) * self.q[s, a] + self.cfg.alpha * td_target

    def end_episode(self) -> None:
        self.eps = max(self.cfg.eps_min, self.eps * self.cfg.eps_decay)

    def load(self, path: str) -> None:
        self.q = np.load(path).astype(np.float32)
        if self.q.shape != (self.n_states, self.n_actions):
            raise ValueError(f"Q-table shape mismatch: expected {(self.n_states, self.n_actions)} got {self.q.shape}")

    def save(self, path: str) -> None:
        np.save(path, self.q)
