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

explore_mode = True            # first flight collects data
explore_steps = 400            # ~8 seconds at ~100Hz
step_count = 0

yaw_scale = 40                 # smoother, safer yaw control
# AUTO-APPROACH (move forward when flower is centered)
approach_enabled = True        # Enable automatic forward movement when centered
approach_threshold = 0.12      # Fractional X error threshold to consider "centered"
approach_speed = 20            # Forward velocity (tello units) when approaching
approach_test_only = True     # Only approach during TEST phase (after training)


# INIT DRONE + YOLO MODEL
myDrone = initializeTello()

print("Loading YOLO-Nano model...")
model = YOLO("yolov8n.pt")

# EXPERIENCE STORAGE & POLICY
# Initialize experience storage with persistence
experience_file = Path("saved_experience") / "experience.pkl"
experience_file.parent.mkdir(parents=True, exist_ok=True)

experience = PILCOExperienceStorage(
    max_size=10000,  # Maximum transitions to store
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
    
    # Check if we have enough data to continue training
    if experience.is_ready_for_training():
        print(f"[POLICY] Continuing with {len(experience)} previous transitions")
        print("[POLICY] Using loaded policy. Press 'r' to retrain with accumulated data.")
        explore_mode = False  # Use loaded policy
    else:
        print("[POLICY] Loaded policy but need more data. Starting exploration.")
        explore_mode = True
else:
    print("[POLICY] No saved policy found. Starting fresh training.")
    explore_mode = True

# Start new flight session
experience.start_new_flight()

# CYCLE SETTINGS
cycle_count = 0  # Number of complete explore→train cycles
max_cycles = None  # None = infinite, set to number to limit cycles
consecutive_cycles_enabled = True  # Enable automatic consecutive explore/train

# Cycle phases: 'explore', 'test', 'wait'
# - 'explore': collect random data
# - 'test': use trained policy
# - 'wait': waiting for user input to start next cycle
current_phase = 'explore'
test_steps = 200  # Steps to test policy before returning to exploration
test_step_count = 0  # Counter for test phase

print("\n" + "="*60)
print("CONTROLS:")
print("  't' - Takeoff (starts EXPLORE phase)")
print("  'l' - Land")
print("  'n' - Skip TEST phase, return to EXPLORE")
print("  'c' - Toggle consecutive cycles")
print("  'r' - Manual retrain PILCO with current experience")
print("  's' - Save current policy manually")
print("  'i' - Show experience statistics")
print("  'q' - Quit and save")
print("="*60)
print(f"Initial mode: {'Exploration' if explore_mode else 'PILCO (loaded policy)'}")
print(f"Consecutive Cycles: {'ENABLED' if consecutive_cycles_enabled else 'DISABLED'}")
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
    if current_phase == 'explore' or explore_mode:
        # Random yaw exploration
        action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
    elif current_phase == 'test':
        # Use trained PILCO policy
        if pilco_policy is not None:
            action = pilco_policy.compute_action(state)  # [-1, 1]
        else:
            # Fallback if no policy yet
            action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
    else:
        # Shouldn't reach here, but safety fallback
        action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

    # -------------------------
    # Convert action → yaw velocity
    # -------------------------
    raw_yaw = action[0] * yaw_scale

    # DEADZONE (remove jitter & drift)
    if abs(raw_yaw) < 20:
        yaw_cmd = 0
    else:
        yaw_cmd = int(raw_yaw)

    # -------------------------
    # Send command (yaw + exploration motion or test-only forward approach)
    # -------------------------
    # Determine linear commands separately from policy-based yaw.
    try:
        exploration_flag = (current_phase == 'explore') or explore_mode

        if exploration_flag:
            # During exploration, allow motion in any horizontal direction (not recorded as action)
            # Keep magnitudes small for safety
            lr_cmd = int(np.random.uniform(-20, 20))   # left/right
            fb_cmd = int(np.random.uniform(-20, 20))   # forward/back
            ud_cmd = 0
        else:
            # Default: no exploration movement
            lr_cmd = 0
            ud_cmd = 0

            # Test phase: allow forward approach when centered
            fb_cmd = 0
            if approach_enabled and area > 0:
                # Move forward as soon as a flower is detected in frame
                # (no longer requires the flower to be centered)
                if (not approach_test_only) or (current_phase == 'test'):
                    fb_cmd = int(approach_speed)

        myDrone.send_rc_control(lr_cmd, fb_cmd, ud_cmd, yaw_cmd)
    except Exception as e:
        print(f"[ERROR] RC control failed: {e}")

    # -------------------------
    # Store transition (s, a, s')
    # -------------------------
    if prev_state is not None:
        experience.add(prev_state, prev_action, state)

    prev_state = state
    prev_action = action
    step_count += 1

    # -------------------------
    # TRAIN PILCO AFTER EXPLORATION
    # -------------------------
    if (current_phase == 'explore' or explore_mode) and step_count >= explore_steps:
        print("\n[POLICY] Exploration complete. Training PILCO model...")
        
        if not experience.is_ready_for_training():
            print(f"[POLICY] Not enough data for training. Need {experience.min_size_for_training}, "
                  f"have {len(experience)}. Continuing exploration...")
        else:
            S, A, S2 = experience.get()
            print(f"[POLICY] Training with {len(S)} transitions from {len(set(experience.flight_ids))} flights")
            
            # If we have a previous model, we could continue training it
            if pilco_model is not None:
                print("[POLICY] Continuing training from previous policy...")
            
            pilco, controller = train_pilco(S, A, S2)
            pilco_model = pilco
            pilco_policy = PILCOControllerWrapper(controller)

            # Save the trained policy automatically
            save_policy(pilco_model, controller, experience)
            print("[POLICY] Policy saved automatically after training.")

            # Free large PILCO GP structures if they are not needed during testing.
            # The controller (wrapped by pilco_policy) is sufficient for action computation.
            try:
                # Remove large GP matrices to reduce memory pressure
                if hasattr(pilco_model, 'gps'):
                    pilco_model.gps = None
                # Remove reference to the full model (we keep controller in pilco_policy)
                pilco_model = None
                gc.collect()
                print("[POLICY] Freed PILCO GP memory before TEST phase.")
            except Exception as e:
                print(f"[POLICY] Warning: could not free PILCO memory: {e}")

            # Transition to test phase
            current_phase = 'test'
            test_step_count = 0
            explore_mode = False
            print(f"[POLICY] PILCO training complete. Switching to TEST phase ({test_steps} steps).")
            print("[POLICY] Use 'n' to skip testing and start next exploration cycle.")
    
    # AUTO-CYCLE: Test phase → back to Exploration
    if current_phase == 'test':
        test_step_count += 1
        
        # After testing for enough steps, automatically return to exploration
        if test_step_count >= test_steps and consecutive_cycles_enabled:
            cycle_count += 1
            print(f"\n[CYCLE #{cycle_count}] Test phase complete ({test_step_count} steps).")
            print(f"[CYCLE #{cycle_count}] Automatic cycle enabled. Returning to EXPLORATION phase.")
            print(f"[CYCLE #{cycle_count}] Press 't' to take off and continue with next exploration...")
            
            current_phase = 'wait'
            explore_mode = True
            step_count = 0

    # OVERLAYS
    cv2.putText(frame, f"yaw: {yaw_cmd}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    phase_color = (0, 255, 0) if current_phase == 'explore' else (255, 0, 0) if current_phase == 'test' else (200, 200, 0)
    phase_text = f"Phase: {current_phase.upper()}"
    if current_phase == 'test':
        phase_text += f" ({test_step_count}/{test_steps})"
    
    cv2.putText(frame, phase_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
    
    # Show experience stats
    exp_stats = experience.get_stats()
    cv2.putText(frame,
                f"Transitions: {exp_stats['total_transitions']} | "
                f"Flights: {exp_stats['num_flights']} | "
                f"Cycles: {cycle_count}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200,200,200), 1)

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

    elif key == ord('t'):
        safe_takeoff(myDrone)
        experience.start_new_flight()  # Mark new flight after takeoff
        step_count = 0  # Reset step counter
        test_step_count = 0  # Reset test counter
        current_phase = 'explore'  # Always start in exploration when taking off
        explore_mode = True
        print(f"[TAKEOFF] Starting EXPLORATION phase (cycle #{cycle_count + 1})")

    elif key == ord('l'):
        try: 
            myDrone.land()
            experience.end_flight()  # Save experience after landing
        except Exception as e: print(f"[ERROR] Land failed: {e}")
    
    elif key == ord('n'):
        # Skip test phase, return to exploration
        if current_phase == 'test':
            cycle_count += 1
            print(f"\n[SKIP] Skipping test phase.")
            print(f"[CYCLE #{cycle_count}] Returning to EXPLORATION phase.")
            current_phase = 'wait'
            explore_mode = True
            step_count = 0
        else:
            print("[SKIP] Can only skip during TEST phase.")
    
    elif key == ord('c'):
        # Toggle consecutive cycles
        consecutive_cycles_enabled = not consecutive_cycles_enabled
        status = "ENABLED" if consecutive_cycles_enabled else "DISABLED"
        print(f"\n[CYCLE] Consecutive cycles: {status}")
        if consecutive_cycles_enabled:
            print("[CYCLE] After test phase, auto-returns to exploration.")
        else:
            print("[CYCLE] Manual control only. Use 'n' to transition phases manually.")
    
    elif key == ord('r'):
        # Retrain PILCO with current experience
        if experience.is_ready_for_training():
            print("\n[POLICY] Retraining PILCO with current experience...")
            S, A, S2 = experience.get()
            print(f"[POLICY] Using {len(S)} transitions from {len(set(experience.flight_ids))} flights")
            
            pilco, controller = train_pilco(S, A, S2)
            pilco_model = pilco
            pilco_policy = PILCOControllerWrapper(controller)
            
            # Save the retrained policy
            save_policy(pilco_model, controller, experience)
            print("[POLICY] Retraining complete. Policy saved.")
            explore_mode = False
        else:
            print(f"[POLICY] Not enough data for retraining. Need {experience.min_size_for_training}, "
                  f"have {len(experience)}")
    
    elif key == ord('s'):
        # Manually save current policy
        if pilco_model is not None and pilco_policy is not None:
            controller = pilco_policy.controller if hasattr(pilco_policy, 'controller') else None
            if controller is not None:
                save_policy(pilco_model, controller, experience)
                print("[POLICY] Policy saved manually.")
            else:
                print("[POLICY] Could not access controller.")
        else:
            print("[POLICY] No policy to save. Train PILCO first.")
    
    elif key == ord('i'):
        # Show experience statistics
        stats = experience.get_stats()
        print("\n" + "="*60)
        print("EXPERIENCE STATISTICS:")
        print(f"  Total transitions: {stats['total_transitions']}")
        print(f"  Number of flights: {stats['num_flights']}")
        print(f"  Buffer usage: {stats['buffer_usage']*100:.1f}%")
        print(f"  Oldest data: {stats['oldest_transition_age_hours']:.2f} hours ago")
        print(f"  Newest data: {stats['newest_transition_age_hours']:.2f} hours ago")
        if stats['transitions_per_flight']:
            print(f"  Transitions per flight: {stats['transitions_per_flight']}")
        print(f"  Completed cycles: {cycle_count}")
        print(f"  Current phase: {current_phase.upper()}")
        print(f"  Consecutive cycles: {'ENABLED' if consecutive_cycles_enabled else 'DISABLED'}")
        print("="*60)
