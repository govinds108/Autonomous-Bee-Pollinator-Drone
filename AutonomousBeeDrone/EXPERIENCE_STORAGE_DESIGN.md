# PILCO Experience Storage System Design

## Overview

The experience storage system is designed to efficiently store and manage (state, action, next_state) transitions for PILCO training, allowing the drone to learn from accumulated experience across multiple flight sessions.

## Data Structure

### Core Components

```python
PILCOExperienceStorage:
    - states: deque of state vectors (numpy arrays, dtype=float32)
    - actions: deque of action vectors (numpy arrays, dtype=float32)
    - next_states: deque of next state vectors (numpy arrays, dtype=float32)
    - timestamps: deque of timestamps (float, seconds since epoch)
    - flight_ids: deque of flight session IDs (int)
```

### Data Format

Each transition is stored as:
- **State**: `np.array([yaw_error], dtype=np.float32)` - shape: (1,)
- **Action**: `np.array([yaw_action], dtype=np.float32)` - shape: (1,)
- **Next State**: `np.array([next_yaw_error], dtype=np.float32)` - shape: (1,)

### Metadata

- `current_flight_id`: Tracks the current flight session
- `total_transitions`: Total number of transitions ever added
- `flight_transition_counts`: Dictionary mapping flight_id → number of transitions

## Adding New Transitions

### After Each Step

```python
# In main loop, after each control step:
if prev_state is not None:
    experience.add(prev_state, prev_action, state)
```

### After Each Flight

```python
# Mark end of flight (automatically saves to disk)
experience.end_flight()
```

### Batch Addition

```python
# Add multiple transitions at once (e.g., after a flight)
experience.add_batch(states_list, actions_list, next_states_list)
```

## Data Removal Strategy

### FIFO (First-In-First-Out) - Default

**When**: Automatically when buffer reaches `max_size`

**How**: Uses `deque` with `maxlen` parameter - oldest data is automatically removed

**Why**: 
- Simple and efficient
- Ensures recent data is always available
- Prevents memory overflow

**Implementation**:
```python
self.states = deque(maxlen=max_size)  # Automatically removes oldest
```

### Random Removal

**When**: When buffer is full and new data arrives

**How**: Randomly selects and removes an old transition

**Why**: 
- Can preserve diverse experiences
- Less bias toward recent data

**Usage**:
```python
experience = PILCOExperienceStorage(removal_strategy='random')
```

### Manual Cleanup

```python
# Keep only recent flights (e.g., last 5 flights)
experience.clear_old_flights(keep_recent=5)
```

## Efficiency Considerations

### Maximum Size Management

**Default**: `max_size=10000` transitions

**Rationale**:
- GP training complexity: O(n³) where n = number of training points
- 10,000 points ≈ reasonable training time (~minutes)
- Balances data richness vs. computational cost

**Adjustment**:
```python
# For faster training (fewer points)
experience = PILCOExperienceStorage(max_size=5000)

# For richer data (more points, slower training)
experience = PILCOExperienceStorage(max_size=20000)
```

### Minimum Training Size

**Default**: `min_size_for_training=50` transitions

**Why**: 
- GP needs minimum data to learn meaningful patterns
- Prevents training on insufficient data

**Check**:
```python
if experience.is_ready_for_training():
    # Safe to train
    S, A, S2 = experience.get()
    pilco, controller = train_pilco(S, A, S2)
```

## Persistence to Disk

### Automatic Saving

- **After each flight**: `experience.end_flight()` saves automatically
- **On program exit**: Main loop saves before quitting

### Manual Saving

```python
experience.save_to_disk()  # Uses default file
experience.save_to_disk("custom_path.pkl")  # Custom path
```

### Loading

```python
# Automatically loads on initialization if file exists
experience = PILCOExperienceStorage(storage_file="saved_experience/experience.pkl")
```

### File Structure

```
saved_experience/
└── experience.pkl  # Contains:
    - states: List[np.ndarray]
    - actions: List[np.ndarray]
    - next_states: List[np.ndarray]
    - timestamps: List[float]
    - flight_ids: List[int]
    - metadata: dict
```

## Usage Examples

### Basic Usage

```python
from pilco_experience_storage import PILCOExperienceStorage

# Initialize
experience = PILCOExperienceStorage(
    max_size=10000,
    min_size_for_training=50,
    storage_file="experience.pkl"
)

# Start flight
experience.start_new_flight()

# Add transitions during flight
for step in range(100):
    state = get_state(...)
    action = choose_action(state)
    next_state = get_state(...)
    experience.add(state, action, next_state)

# End flight (auto-saves)
experience.end_flight()
```

### Training with Accumulated Data

```python
# Check if ready
if experience.is_ready_for_training():
    # Get all transitions
    S, A, S2 = experience.get()
    
    # Train PILCO
    pilco, controller = train_pilco(S, A, S2)
    
    # Save policy
    save_policy(pilco, controller, experience)
```

### Getting Statistics

```python
stats = experience.get_stats()
print(f"Total transitions: {stats['total_transitions']}")
print(f"Number of flights: {stats['num_flights']}")
print(f"Buffer usage: {stats['buffer_usage']*100:.1f}%")
```

## Integration with PILCO Training

### Data Preparation

```python
# Get all transitions
S, A, S2 = experience.get()

# PILCO expects:
# X = states (except last)
# U = actions (except last)
# Y = state_deltas (S2 - S, except last)

def prepare_pilco_data(S, A, S2):
    X = S[:-1]
    U = A[:-1]
    Y = (S2 - S)[:-1]
    return X, U, Y
```

### Training Workflow

1. **Collect Data**: Add transitions during exploration flights
2. **Check Readiness**: `experience.is_ready_for_training()`
3. **Get Data**: `S, A, S2 = experience.get()`
4. **Train**: `pilco, controller = train_pilco(S, A, S2)`
5. **Save Policy**: `save_policy(pilco, controller, experience)`

## Performance Considerations

### Memory Usage

- **Per transition**: ~100 bytes (3 arrays × ~30 bytes)
- **10,000 transitions**: ~1 MB
- **100,000 transitions**: ~10 MB

### Training Time

- **1,000 points**: ~10 seconds
- **5,000 points**: ~2 minutes
- **10,000 points**: ~8 minutes
- **20,000 points**: ~30+ minutes

### Recommendations

- **For development**: `max_size=5000` (faster iteration)
- **For production**: `max_size=10000` (good balance)
- **For research**: `max_size=20000` (maximum data, slower training)

## Best Practices

1. **Save frequently**: Use `experience.end_flight()` after each flight
2. **Monitor buffer**: Check `experience.get_stats()` regularly
3. **Clean old data**: Use `clear_old_flights()` if buffer gets too large
4. **Validate data**: Check `is_ready_for_training()` before training
5. **Backup policies**: Keep multiple policy checkpoints

## Troubleshooting

### Buffer Full

```python
# Check usage
stats = experience.get_stats()
if stats['buffer_usage'] > 0.9:
    # Clear old flights
    experience.clear_old_flights(keep_recent=5)
```

### Training Too Slow

```python
# Reduce buffer size
experience = PILCOExperienceStorage(max_size=5000)

# Or use recent data only
S, A, S2 = experience.get_recent(5000)
```

### Not Enough Data

```python
# Check current size
if not experience.is_ready_for_training():
    print(f"Need {experience.min_size_for_training} transitions, "
          f"have {len(experience)}")
    # Continue collecting data
```

