"""
Experience Storage System for PILCO
====================================

This module provides a comprehensive experience storage system for PILCO that:
- Stores (state, action, next_state) transitions efficiently
- Appends new transitions after each flight
- Manages data removal to keep GP training efficient
- Supports persistence to disk for long-term learning
"""

import numpy as np
import pickle
import os
from pathlib import Path
from collections import deque
from typing import Tuple, Optional, List
import time


class PILCOExperienceStorage:
    """
    Advanced experience storage for PILCO with efficient data management.
    
    Data Structure:
    - states: List of state vectors (numpy arrays)
    - actions: List of action vectors (numpy arrays)  
    - next_states: List of next state vectors (numpy arrays)
    - timestamps: List of timestamps for each transition
    - flight_ids: List of flight session IDs
    
    The storage automatically manages data to keep GP training efficient:
    - Maintains a maximum size limit
    - Can prioritize recent or diverse data
    - Supports FIFO (oldest first) or importance-based removal
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 min_size_for_training: int = 50,
                 storage_file: Optional[str] = None,
                 removal_strategy: str = 'fifo'):
        """
        Initialize experience storage.
        
        Args:
            max_size: Maximum number of transitions to store
            min_size_for_training: Minimum transitions needed for training
            storage_file: Path to persistent storage file (None = no persistence)
            removal_strategy: 'fifo' (oldest first) or 'random' (random removal)
        """
        self.max_size = max_size
        self.min_size_for_training = min_size_for_training
        self.storage_file = storage_file
        self.removal_strategy = removal_strategy
        
        # Core data structures
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.flight_ids = deque(maxlen=max_size)
        
        # Metadata
        self.current_flight_id = 0
        self.total_transitions = 0
        self.flight_transition_counts = {}  # Track transitions per flight
        
        # Load from disk if file exists
        if storage_file and os.path.exists(storage_file):
            self.load_from_disk()
            print(f"[EXPERIENCE] Loaded {len(self.states)} transitions from {storage_file}")
        else:
            print(f"[EXPERIENCE] Initialized new experience storage (max_size={max_size})")
    
    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """
        Add a new transition (s, a, s') to the storage.
        
        Args:
            state: Current state vector
            action: Action taken
            next_state: Resulting next state
        """
        # Convert to numpy arrays and ensure correct types
        state = np.array(state, dtype=np.float32).flatten()
        action = np.array(action, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        
        # Check if we need to remove old data
        if len(self.states) >= self.max_size:
            self._remove_old_data()
        
        # Add new transition
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.next_states.append(next_state.copy())
        self.timestamps.append(time.time())
        self.flight_ids.append(self.current_flight_id)
        
        # Update metadata
        self.total_transitions += 1
        if self.current_flight_id not in self.flight_transition_counts:
            self.flight_transition_counts[self.current_flight_id] = 0
        self.flight_transition_counts[self.current_flight_id] += 1
    
    def add_batch(self, states: List[np.ndarray], actions: List[np.ndarray], 
                  next_states: List[np.ndarray]):
        """
        Add multiple transitions at once (e.g., after a flight).
        
        Args:
            states: List of state vectors
            actions: List of action vectors
            next_states: List of next state vectors
        """
        assert len(states) == len(actions) == len(next_states), \
            "All lists must have the same length"
        
        for s, a, s_next in zip(states, actions, next_states):
            self.add(s, a, s_next)
    
    def start_new_flight(self):
        """Mark the start of a new flight session."""
        self.current_flight_id += 1
        print(f"[EXPERIENCE] Started flight session {self.current_flight_id}")
    
    def end_flight(self):
        """Mark the end of current flight and optionally save to disk."""
        flight_count = self.flight_transition_counts.get(self.current_flight_id, 0)
        print(f"[EXPERIENCE] Flight {self.current_flight_id} ended with {flight_count} transitions")
        
        # Auto-save after each flight
        if self.storage_file:
            self.save_to_disk()
            print(f"[EXPERIENCE] Saved {len(self.states)} total transitions to disk")
    
    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all stored transitions as numpy arrays.
        
        Returns:
            Tuple of (states, actions, next_states) as numpy arrays
        """
        if len(self.states) == 0:
            return np.array([]), np.array([]), np.array([])
        
        S = np.array(self.states, dtype=np.float32)
        A = np.array(self.actions, dtype=np.float32)
        S2 = np.array(self.next_states, dtype=np.float32)
        
        return S, A, S2
    
    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the n most recent transitions.
        
        Args:
            n: Number of recent transitions to return
        
        Returns:
            Tuple of (states, actions, next_states) for recent transitions
        """
        if len(self.states) == 0:
            return np.array([]), np.array([]), np.array([])
        
        n = min(n, len(self.states))
        S = np.array(list(self.states)[-n:], dtype=np.float32)
        A = np.array(list(self.actions)[-n:], dtype=np.float32)
        S2 = np.array(list(self.next_states)[-n:], dtype=np.float32)
        
        return S, A, S2
    
    def get_from_flight(self, flight_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all transitions from a specific flight.
        
        Args:
            flight_id: Flight session ID
        
        Returns:
            Tuple of (states, actions, next_states) for that flight
        """
        indices = [i for i, fid in enumerate(self.flight_ids) if fid == flight_id]
        
        if not indices:
            return np.array([]), np.array([]), np.array([])
        
        S = np.array([self.states[i] for i in indices], dtype=np.float32)
        A = np.array([self.actions[i] for i in indices], dtype=np.float32)
        S2 = np.array([self.next_states[i] for i in indices], dtype=np.float32)
        
        return S, A, S2
    
    def _remove_old_data(self):
        """Remove old data according to removal strategy."""
        if self.removal_strategy == 'fifo':
            # FIFO: Oldest data is automatically removed by deque maxlen
            # This happens automatically, but we can add logging
            if len(self.states) >= self.max_size:
                print(f"[EXPERIENCE] Buffer full ({len(self.states)}/{self.max_size}), "
                      f"removing oldest transitions (FIFO)")
        elif self.removal_strategy == 'random':
            # Random: Remove a random old transition
            if len(self.states) > 0:
                idx = np.random.randint(0, len(self.states))
                # Convert to list, remove, convert back
                states_list = list(self.states)
                actions_list = list(self.actions)
                next_states_list = list(self.next_states)
                timestamps_list = list(self.timestamps)
                flight_ids_list = list(self.flight_ids)
                
                del states_list[idx]
                del actions_list[idx]
                del next_states_list[idx]
                del timestamps_list[idx]
                del flight_ids_list[idx]
                
                self.states = deque(states_list, maxlen=self.max_size)
                self.actions = deque(actions_list, maxlen=self.max_size)
                self.next_states = deque(next_states_list, maxlen=self.max_size)
                self.timestamps = deque(timestamps_list, maxlen=self.max_size)
                self.flight_ids = deque(flight_ids_list, maxlen=self.max_size)
    
    def clear_old_flights(self, keep_recent: int = 5):
        """
        Remove data from old flights, keeping only recent ones.
        
        Args:
            keep_recent: Number of recent flights to keep
        """
        if len(self.flight_ids) == 0:
            return
        
        # Find which flight IDs to keep
        unique_flights = sorted(set(self.flight_ids), reverse=True)
        flights_to_keep = set(unique_flights[:keep_recent])
        
        # Filter data
        states_list = []
        actions_list = []
        next_states_list = []
        timestamps_list = []
        flight_ids_list = []
        
        for i, fid in enumerate(self.flight_ids):
            if fid in flights_to_keep:
                states_list.append(self.states[i])
                actions_list.append(self.actions[i])
                next_states_list.append(self.next_states[i])
                timestamps_list.append(self.timestamps[i])
                flight_ids_list.append(fid)
        
        # Reconstruct deques
        self.states = deque(states_list, maxlen=self.max_size)
        self.actions = deque(actions_list, maxlen=self.max_size)
        self.next_states = deque(next_states_list, maxlen=self.max_size)
        self.timestamps = deque(timestamps_list, maxlen=self.max_size)
        self.flight_ids = deque(flight_ids_list, maxlen=self.max_size)
        
        print(f"[EXPERIENCE] Kept data from {len(flights_to_keep)} recent flights, "
              f"removed older flights. Total transitions: {len(self.states)}")
    
    def save_to_disk(self, file_path: Optional[str] = None):
        """
        Save experience data to disk.
        
        Args:
            file_path: Path to save file (uses self.storage_file if None)
        """
        if file_path is None:
            file_path = self.storage_file
        
        if file_path is None:
            print("[EXPERIENCE] No storage file specified, skipping save")
            return
        
        # Create directory if needed
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
            'removal_strategy': self.removal_strategy
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[EXPERIENCE] Saved {len(self.states)} transitions to {file_path}")
    
    def load_from_disk(self, file_path: Optional[str] = None):
        """
        Load experience data from disk.
        
        Args:
            file_path: Path to load file (uses self.storage_file if None)
        """
        if file_path is None:
            file_path = self.storage_file
        
        if file_path is None or not os.path.exists(file_path):
            print(f"[EXPERIENCE] No file to load from: {file_path}")
            return
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore data
        self.states = deque(data['states'], maxlen=data.get('max_size', self.max_size))
        self.actions = deque(data['actions'], maxlen=data.get('max_size', self.max_size))
        self.next_states = deque(data['next_states'], maxlen=data.get('max_size', self.max_size))
        self.timestamps = deque(data['timestamps'], maxlen=data.get('max_size', self.max_size))
        self.flight_ids = deque(data['flight_ids'], maxlen=data.get('max_size', self.max_size))
        
        # Restore metadata
        self.current_flight_id = data.get('current_flight_id', 0)
        self.total_transitions = data.get('total_transitions', len(self.states))
        self.flight_transition_counts = data.get('flight_transition_counts', {})
        
        # Update max_size if different
        if 'max_size' in data:
            self.max_size = data['max_size']
        
        print(f"[EXPERIENCE] Loaded {len(self.states)} transitions from {file_path}")
    
    def get_stats(self) -> dict:
        """Get statistics about stored experience."""
        if len(self.states) == 0:
            return {
                'total_transitions': 0,
                'num_flights': 0,
                'buffer_usage': 0.0,
                'oldest_transition_age_hours': 0.0,
                'newest_transition_age_hours': 0.0
            }
        
        current_time = time.time()
        oldest_age = (current_time - min(self.timestamps)) / 3600.0
        newest_age = (current_time - max(self.timestamps)) / 3600.0
        
        return {
            'total_transitions': len(self.states),
            'num_flights': len(set(self.flight_ids)),
            'buffer_usage': len(self.states) / self.max_size,
            'oldest_transition_age_hours': oldest_age,
            'newest_transition_age_hours': newest_age,
            'transitions_per_flight': self.flight_transition_counts
        }
    
    def is_ready_for_training(self) -> bool:
        """Check if there's enough data for training."""
        return len(self.states) >= self.min_size_for_training
    
    def __len__(self):
        """Return number of stored transitions."""
        return len(self.states)
    
    def __repr__(self):
        return (f"PILCOExperienceStorage("
                f"transitions={len(self.states)}, "
                f"flights={len(set(self.flight_ids))}, "
                f"max_size={self.max_size})")


# Backward compatibility: Alias for old name
ReplayBufferPILCO = PILCOExperienceStorage

