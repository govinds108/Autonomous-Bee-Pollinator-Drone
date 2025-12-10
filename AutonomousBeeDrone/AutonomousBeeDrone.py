from utils import *
from pilco_experience_storage import PILCOExperienceStorage
from pilco_training import train_pilco
from pilco_controller_wrapper import PILCOControllerWrapper
from policy_persistence import save_policy, load_policy

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import gc


# SETTINGS
w, h = 720, 440

explore_steps = 400            # ~8 seconds at ~100Hz
step_count = 0

yaw_scale = 40                 # smoother, safer yaw control
test_forward_speed = 15       # slow forward speed during test phase (adjustable)
# Exploration movement (stronger than test): forward/back and left/right max speeds
explore_forward_speed_max = 30  # reduced for less aggressive exploration
explore_lr_speed_max = 30


# INIT DRONE + YOLO MODEL
myDrone = initializeTello()

print("Loading YOLO-Nano model...")
model = YOLO("yolov8n.pt")

# EXPERIENCE STORAGE & POLICY
# Initialize experience storage with persistence
experience_file = Path("saved_experience") / "experience.pkl"
experience_file.parent.mkdir(parents=True, exist_ok=True)

experience = PILCOExperienceStorage(
    max_size=50000,  # Increased buffer to store more diverse transitions
    min_size_for_training=50,  # Minimum needed for training
    storage_file=str(experience_file),
    removal_strategy='fifo'  # Remove oldest when full
)

pilco_policy = None
pilco_model = None  # Store full PILCO model for saving

prev_state = None
prev_action = None

# Try to load previously saved policy
print("\n[POLICY] Attempting to load saved policy...")
loaded_pilco, loaded_controller, experience_metadata = load_policy()

if loaded_pilco is not None:
    # Policy loaded successfully - continue from previous training
    pilco_model = loaded_pilco
    pilco_policy = PILCOControllerWrapper(loaded_controller)
    print(f"[POLICY] Loaded policy with {len(experience)} transitions")
else:
    print("[POLICY] No saved policy found. Train first with exploration.")

# Start new flight session
experience.start_new_flight()

# Cycle phases: 'explore', 'test', 'wait'
# - 'explore': collect random data
# - 'test': use trained policy
# - 'wait': waiting for user input to start next cycle
current_phase = 'wait'  # Start in waiting state
test_step_count = 0  # Counter for test phase
training_in_progress = False  # True while 'r' training is running

print("\n" + "="*60)
print("CONTROLS:")
print("  'm' - Takeoff (drone starts in WAIT state)")
print("  'e' - Start EXPLORATION (collect random data)")
print("  'r' - (disabled) Train on-drone — use offline script instead")
print("  't' - Start TESTING (run trained policy)")
print("  'l' - Land")
print("  'q' - Quit and save")
print("="*60)
print(f"Experience: {len(experience)} transitions from {len(set(experience.flight_ids))} flights")


# SAFE TAKEOFF FUNCTION (fixed)
def safe_takeoff(drone):
    """
    A reliable takeoff routine:
    - Battery check
    - RC reset
    - Single takeoff call
    """

    try:
        # Battery check
        try:
            batt = drone.get_battery()
            print(f"[TAKEOFF] Battery: {batt}%")
            if batt < 15:
                print("[TAKEOFF] ERROR: Battery too low (<15%).")
                return False
        except Exception as e:
            print(f"[TAKEOFF] WARNING: Could not check battery: {e}")

        # Reset RC control before takeoff
        try:
            drone.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            print(f"[TAKEOFF] WARNING: Could not reset RC control: {e}")
        
        time.sleep(0.1)

        print("[TAKEOFF] Attempting takeoff...")
        drone.takeoff()
        time.sleep(1.2)

        print("[TAKEOFF] SUCCESS!")
        return True

    except Exception as e:
        print(f"[TAKEOFF] CRITICAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# MAIN LOOP
# ============================================================
while True:

    # -------------------------
    # Get camera frame
    # -------------------------
    frame = telloGetFrame(myDrone, w, h)

    # -------------------------
    # Detect flower
    # -------------------------
    frame, info = detectFlower(frame, model)
    (cx, cy), area, conf, flower_score = info

    # -------------------------
    # Compute state (YAW ONLY)
    # -------------------------
    state = get_state(cx, cy, area, w, h)  # shape: (1,)

    # -------------------------
    # CHOOSE ACTION BASED ON PHASE
    # -------------------------
    if current_phase == 'explore':
        # Enhanced exploration with structured forward noise
        # This improves learning by:
        # - Exploring full [-1, 1] action space (matches test phase range)
        # - Adding Gaussian noise for diverse exploration
        # - Maintaining bias toward forward motion for safety
        
        yaw_rand = np.random.uniform(-1.0, 1.0)
        
        # Structured forward exploration with noise
        # Base: random forward movement biased toward positive values
        base_forward = np.random.uniform(0.0, 1.0)  # Mostly forward [0, 1]
        # Add Gaussian noise to explore different forward speeds
        forward_noise = np.random.normal(0, 0.2)  # Mean=0, Std=0.2
        # Combine: creates full [-1, 1] exploration while staying mostly forward
        forward_with_noise = np.clip(base_forward + forward_noise, -1.0, 1.0)
        
        action = np.array([yaw_rand, forward_with_noise], dtype=np.float32)
    elif current_phase == 'test':
        # Use trained PILCO policy
        if pilco_policy is not None:
            action = pilco_policy.compute_action(state)  # [-1, 1]
        else:
            # Fallback if no policy yet
            action = np.array([np.random.uniform(-1.0, 1.0), 0.0], dtype=np.float32)
    else:
        # Wait phase: no action
        action = np.array([0.0, 0.0], dtype=np.float32)

    # -------------------------
    # Convert action → yaw velocity
    # -------------------------
    raw_yaw = float(action[0]) * yaw_scale

    # DEADZONE (remove jitter & drift)
    if abs(raw_yaw) < 20:
        yaw_cmd = 0
    else:
        yaw_cmd = int(raw_yaw)

    # -------------------------
    # Send command (left_right, forward/back, up_down, yaw)
    # - During TEST phase: only move forward when YOLO detects a flower above threshold
    # - During EXPLORE phase: apply stronger random forward/left-right motions
    # - WAIT phase: no translational movement
    # -------------------------
    try:
        left_cmd = 0
        forward_cmd = 0

        if current_phase == 'explore':
            # Random left/right bursts for exploration (no forward motion here),
            # but allow a small random forward command so forward behavior is explored.
            left_cmd = int(np.random.randint(-explore_lr_speed_max, explore_lr_speed_max + 1))
            forward_cmd = int(action[1] * explore_forward_speed_max)

        elif current_phase == 'test':
            # FORWARD is now controlled by the learned policy (action[1]).
            # Map action[1] ∈ [-1,1] to forward speed; clip to safe range.
            forward_cmd = int(np.clip(float(action[1]), -1.0, 1.0) * test_forward_speed)

        # send_rc_control(left_right, forward_back, up_down, yaw)
        myDrone.send_rc_control(left_cmd, forward_cmd, 0, yaw_cmd)
    except Exception as e:
        print(f"[ERROR] RC control failed: {e}")

    # -------------------------
    # Store transition (s, a, s') - during exploration and testing only
    # -------------------------
    if prev_state is not None and current_phase in ['explore', 'test']:
        experience.add(prev_state, prev_action, state)

    prev_state = state
    prev_action = action
    step_count += 1

    # -------------------------
    # TRAINING IS MANUAL - triggered by keyboard command (handled in keyboard input section)
    # -------------------------

    # OVERLAYS
    cv2.putText(frame, f"yaw: {yaw_cmd}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    phase_color = (0, 255, 0) if current_phase == 'explore' else \
                  (0, 165, 255) if current_phase == 'training' else \
                  (255, 0, 0) if current_phase == 'test' else (200, 200, 0)
    # Phase / mode overlay
    if training_in_progress:
        phase_text = "Phase: TRAINING"
        phase_color = (0, 140, 255)
    else:
        phase_color = (0, 255, 0) if current_phase == 'explore' else (255, 0, 0) if current_phase == 'test' else (200, 200, 0)
        phase_text = f"Phase: {current_phase.upper()}"
    
    cv2.putText(frame, phase_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)

    # Show experience stats
    exp_stats = experience.get_stats()
    cv2.putText(frame, f"Transitions: {exp_stats['total_transitions']} | Flights: {exp_stats['num_flights']}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Controls overlay (right side)
    controls = [
        "Controls:",
        "m: Takeoff",
        "l: Land",
        "e: Exploration (collect)",
        "t: Test (run policy)",
        "q: Quit"
    ]

    cx_text = w - 10
    cy = 20
    for line in controls:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, line, (cx_text - tw, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        cy += int(th * 1.6)

    cv2.imshow("Autonomous Bee Drone (PILCO)", frame)

    # KEYBOARD INPUT
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Save everything before quitting
        print("\n[POLICY] Saving before exit...")
        experience.end_flight()  # Saves experience
        if pilco_model is not None and pilco_policy is not None:
            controller = pilco_policy.controller if hasattr(pilco_policy, 'controller') else None
            if controller is not None:
                save_policy(pilco_model, controller, experience)
        try: myDrone.land()
        except Exception as e: print(f"[ERROR] Land failed: {e}")
        break

    elif key == ord('m'):
        # Takeoff
        safe_takeoff(myDrone)
        experience.start_new_flight()  # Mark new flight after takeoff
        step_count = 0  # Reset step counter
        current_phase = 'wait'  # Start in waiting state
        print("[TAKEOFF] SUCCESS! Drone in WAIT state. Press 'e' for exploration or 't' for testing.")

    elif key == ord('l'):
        # Land
        try: 
            myDrone.land()
            experience.end_flight()  # Save experience after landing
            print("[LAND] Landed successfully.")
        except Exception as e: print(f"[ERROR] Land failed: {e}")
    
    elif key == ord('e'):
        # Start exploration immediately (no WAIT requirement)
        current_phase = 'explore'
        step_count = 0
        print("[EXPLORE] Starting exploration. Flying randomly and collecting data...")

    elif key == ord('r'):
        # On-drone training has been disabled. Save current flight experience
        # and inform user to run the offline training script instead.
        try:
            experience.end_flight()
            print("[POLICY] Saved current experience to disk.")
        except Exception as e:
            print(f"[POLICY] Warning: could not save experience: {e}")

        print("[POLICY] On-drone training is disabled. Run `python train_offline.py` to train from saved experience.")
        # Ensure state remains safe
        training_in_progress = False
        current_phase = 'wait'

    elif key == ord('t'):
        # Start testing immediately (no WAIT requirement)
        if pilco_policy is not None:
            current_phase = 'test'
            test_step_count = 0
            print("[TEST] Starting testing with trained policy...")
        else:
            print("[TEST] No trained policy available. Train with exploration first ('r')")

