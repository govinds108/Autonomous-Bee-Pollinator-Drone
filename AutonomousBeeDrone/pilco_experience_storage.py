import numpy as np
import pickle
import os
from pathlib import Path
from collections import deque
from typing import Tuple, Optional, List
import time


class PILCOExperienceStorage:
    """
    Experience buffer for PILCO.
    
    Stores (state, action, next_state) transitions with metadata
    and supports persistence and basic data management.
    """

    def __init__(self,
                 max_size: int = 10000,
                 min_size_for_training: int = 50,
                 storage_file: Optional[str] = None,
                 removal_strategy: str = 'fifo'):
        self.max_size = max_size
        self.min_size_for_training = min_size_for_training
        self.storage_file = storage_file
        self.removal_strategy = removal_strategy

        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.flight_ids = deque(maxlen=max_size)

        self.current_flight_id = 0
        self.total_transitions = 0
        self.flight_transition_counts = {}

        if storage_file and os.path.exists(storage_file):
            self.load_from_disk()
            print(f"[EXPERIENCE] Loaded {len(self.states)} transitions from {storage_file}")
        else:
            print(f"[EXPERIENCE] Initialized empty buffer (max_size={max_size})")

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        state = np.array(state, dtype=np.float32).flatten()
        action = np.array(action, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()

        if len(self.states) >= self.max_size:
            self._remove_old_data()

        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.next_states.append(next_state.copy())
        self.timestamps.append(time.time())
        self.flight_ids.append(self.current_flight_id)

        self.total_transitions += 1
        self.flight_transition_counts.setdefault(self.current_flight_id, 0)
        self.flight_transition_counts[self.current_flight_id] += 1

    def add_batch(self, states: List[np.ndarray],
                  actions: List[np.ndarray],
                  next_states: List[np.ndarray]):
        for s, a, s_next in zip(states, actions, next_states):
            self.add(s, a, s_next)

    def start_new_flight(self):
        self.current_flight_id += 1
        print(f"[EXPERIENCE] Started flight {self.current_flight_id}")

    def end_flight(self):
        count = self.flight_transition_counts.get(self.current_flight_id, 0)
        print(f"[EXPERIENCE] Flight {self.current_flight_id} ended ({count} transitions)")

        if self.storage_file:
            self.save_to_disk()
            print(f"[EXPERIENCE] Saved {len(self.states)} transitions")

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.states:
            return np.array([]), np.array([]), np.array([])

        def to_vec(x):
            return np.array(x, dtype=np.float32).reshape(-1)

        states = [to_vec(s) for s in self.states]
        actions = [to_vec(a) for a in self.actions]
        next_states = [to_vec(s2) for s2 in self.next_states]

        def stack(arr):
            lengths = [a.shape[0] for a in arr]
            if len(set(lengths)) == 1:
                return np.vstack(arr)
            max_len = max(lengths)
            out = np.zeros((len(arr), max_len), dtype=np.float32)
            for i, a in enumerate(arr):
                out[i, : a.shape[0]] = a
            return out

        return stack(states), stack(actions), stack(next_states)

    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.states:
            return np.array([]), np.array([]), np.array([])

        n = min(n, len(self.states))

        states = list(self.states)[-n:]
        actions = list(self.actions)[-n:]
        next_states = list(self.next_states)[-n:]

        def to_vec(x):
            return np.array(x, dtype=np.float32).reshape(-1)

        states = [to_vec(s) for s in states]
        actions = [to_vec(a) for a in actions]
        next_states = [to_vec(s2) for s2 in next_states]

        def stack(arr):
            lengths = [a.shape[0] for a in arr]
            if len(set(lengths)) == 1:
                return np.vstack(arr)
            max_len = max(lengths)
            out = np.zeros((len(arr), max_len), dtype=np.float32)
            for i, a in enumerate(arr):
                out[i, : a.shape[0]] = a
            return out

        return stack(states), stack(actions), stack(next_states)

    def get_from_flight(self, flight_id: int):
        idxs = [i for i, f in enumerate(self.flight_ids) if f == flight_id]
        if not idxs:
            return np.array([]), np.array([]), np.array([])

        def to_vec(x):
            return np.array(x, dtype=np.float32).reshape(-1)

        S = [to_vec(self.states[i]) for i in idxs]
        A = [to_vec(self.actions[i]) for i in idxs]
        S2 = [to_vec(self.next_states[i]) for i in idxs]

        def stack(arr):
            lengths = [a.shape[0] for a in arr]
            if len(set(lengths)) == 1:
                return np.vstack(arr)
            max_len = max(lengths)
            out = np.zeros((len(arr), max_len), dtype=np.float32)
            for i, a in enumerate(arr):
                out[i, : a.shape[0]] = a
            return out

        return stack(S), stack(A), stack(S2)

    def _remove_old_data(self):
        if self.removal_strategy == 'fifo':
            # deque handles removal automatically
            pass
        elif self.removal_strategy == 'random' and len(self.states) > 0:
            idx = np.random.randint(0, len(self.states))
            for q in (self.states, self.actions, self.next_states,
                      self.timestamps, self.flight_ids):
                lst = list(q)
                del lst[idx]
                q.clear()
                q.extend(lst)

    def clear_old_flights(self, keep_recent: int = 5):
        if not self.flight_ids:
            return

        unique = sorted(set(self.flight_ids), reverse=True)
        keep = set(unique[:keep_recent])

        def filter_deque(data, ids):
            out = [d for d, f in zip(data, ids) if f in keep]
            return deque(out, maxlen=self.max_size)

        self.states = filter_deque(self.states, self.flight_ids)
        self.actions = filter_deque(self.actions, self.flight_ids)
        self.next_states = filter_deque(self.next_states, self.flight_ids)
        self.timestamps = filter_deque(self.timestamps, self.flight_ids)
        self.flight_ids = filter_deque(self.flight_ids, self.flight_ids)

        print(f"[EXPERIENCE] Pruned to {len(keep)} flights. Total: {len(self.states)}")

    def save_to_disk(self, file_path: Optional[str] = None):
        file_path = file_path or self.storage_file
        if not file_path:
            return

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'states': list(self.states),
            'actions': list(self.actions),
            'next_states': list(self.next_states),
            'timestamps': list(self.timestamps),
            'flight_ids': list(self.flight_ids),
            'current_flight_id': self.current_flight_id,
            'total_transitions': self.total_transitions,
            'flight_transition_counts': self.flight_transition_counts,
            'max_size': self.max_size,
            'min_size_for_training': self.min_size_for_training,
            'removal_strategy': self.removal_strategy,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_from_disk(self, file_path: Optional[str] = None):
        file_path = file_path or self.storage_file
        if not file_path or not os.path.exists(file_path):
            return

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.states = deque(data['states'], maxlen=data['max_size'])
        self.actions = deque(data['actions'], maxlen=data['max_size'])
        self.next_states = deque(data['next_states'], maxlen=data['max_size'])
        self.timestamps = deque(data['timestamps'], maxlen=data['max_size'])
        self.flight_ids = deque(data['flight_ids'], maxlen=data['max_size'])

        self.current_flight_id = data['current_flight_id']
        self.total_transitions = data['total_transitions']
        self.flight_transition_counts = data['flight_transition_counts']
        self.max_size = data['max_size']

    def get_stats(self) -> dict:
        if not self.states:
            return {
                'total_transitions': 0,
                'num_flights': 0,
                'buffer_usage': 0.0,
                'oldest_transition_age_hours': 0.0,
                'newest_transition_age_hours': 0.0,
            }

        now = time.time()
        oldest = (now - min(self.timestamps)) / 3600
        newest = (now - max(self.timestamps)) / 3600

        return {
            'total_transitions': len(self.states),
            'num_flights': len(set(self.flight_ids)),
            'buffer_usage': len(self.states) / self.max_size,
            'oldest_transition_age_hours': oldest,
            'newest_transition_age_hours': newest,
            'transitions_per_flight': self.flight_transition_counts,
        }

    def is_ready_for_training(self) -> bool:
        return len(self.states) >= self.min_size_for_training

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return (f"PILCOExperienceStorage("
                f"transitions={len(self.states)}, "
                f"flights={len(set(self.flight_ids))}, "
                f"max_size={self.max_size})")


ReplayBufferPILCO = PILCOExperienceStorage
