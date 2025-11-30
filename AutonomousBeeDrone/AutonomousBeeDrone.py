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
explore_forward_speed_max = 40
explore_lr_speed_max = 30
# YOLO confidence threshold required to start moving forward during test
flower_confidence_threshold = 0.35


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
print("  'r' - TRAIN PILCO (train policy from collected data)")
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
        # Wait phase: no action
        action = np.array([0.0], dtype=np.float32)

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
    # Send command (left_right, forward/back, up_down, yaw)
    # - During TEST phase: only move forward when YOLO detects a flower above threshold
    # - During EXPLORE phase: apply stronger random forward/left-right motions
    # - WAIT phase: no translational movement
    # -------------------------
    try:
        left_cmd = 0
        forward_cmd = 0

        if current_phase == 'explore':
            # Random left/right bursts for exploration (no forward motion).
            # We avoid forward/back movement during exploration so the agent focuses on yaw control.
            left_cmd = int(np.random.randint(-explore_lr_speed_max, explore_lr_speed_max + 1))
            forward_cmd = 0

        elif current_phase == 'test':
            # Only move forward if YOLO found a flower with sufficient confidence
            try:
                if conf is not None and conf >= flower_confidence_threshold:
                    forward_cmd = int(test_forward_speed)
            except Exception:
                # If `conf` isn't available for any reason, keep forward_cmd = 0
                forward_cmd = 0

        # send_rc_control(left_right, forward_back, up_down, yaw)
        myDrone.send_rc_control(left_cmd, forward_cmd, 0, yaw_cmd)
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
        "r: Train (fit PILCO)",
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
        # Retrain PILCO with collected experience (can be triggered anytime)
        # Show TRAINING overlay immediately so user sees feedback
        training_in_progress = True
        current_phase = 'training'
        # show immediate training overlay frame so user sees the phase change
        try:
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, "Phase: TRAINING", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
            cv2.putText(overlay_frame, "Training PILCO...", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 1)
            cv2.imshow("Autonomous Bee Drone (PILCO)", overlay_frame)
            cv2.waitKey(1)
        except Exception:
            pass

        if experience.is_ready_for_training():
            print("\n[POLICY] Training PILCO with collected experience...")

            # Run training inside try/finally so we always clear the flag
            try:
                S, A, S2 = experience.get()
                print(f"[POLICY] Training with {len(S)} transitions from {len(set(experience.flight_ids))} flights")

                pilco, controller = train_pilco(S, A, S2)
                pilco_model = pilco
                pilco_policy = PILCOControllerWrapper(controller)

                # Save the trained policy
                save_policy(pilco_model, controller, experience)
                print("[POLICY] Policy trained and saved!")

                # Free large PILCO GP structures
                try:
                    if hasattr(pilco_model, 'gps'):
                        pilco_model.gps = None
                    pilco_model = None
                    gc.collect()
                    print("[POLICY] Freed PILCO GP memory after training.")
                except Exception as e:
                    print(f"[POLICY] Warning: could not free PILCO memory: {e}")

                # After training, return to WAIT state (manual control)
                current_phase = 'wait'
                print("[POLICY] Training complete. Use 't' to start testing or 'e' to resume exploration.")
            finally:
                training_in_progress = False
        else:
            print(f"[POLICY] Not enough data for training. Need {experience.min_size_for_training}, "
                  f"have {len(experience)}. Continue exploring.")
            # Give a brief overlay showing not enough data, then revert
            try:
                overlay_frame = frame.copy()
                cv2.putText(overlay_frame, "Phase: TRAINING", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                cv2.putText(overlay_frame, "Not enough data to train", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 1)
                cv2.imshow("Autonomous Bee Drone (PILCO)", overlay_frame)
                cv2.waitKey(500)
            except Exception:
                pass
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

