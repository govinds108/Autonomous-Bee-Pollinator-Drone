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


# Basic runtime settings
w, h = 720, 440            # camera resolution
explore_steps = 400        # ~8 seconds of exploration
step_count = 0

# Action scaling / speeds
yaw_scale = 40             # keeps the drone from spinning too aggressively
test_forward_speed = 15    # slow forward speed when testing the learned policy
explore_forward_speed_max = 40   # stronger random movement for data collection
explore_lr_speed_max = 30        # lateral exploration strength

flower_confidence_threshold = 0.35   # YOLO threshold for allowing forward movement


# Initialize drone + detector
myDrone = initializeTello()

print("Loading YOLO-Nano model...")
model = YOLO("yolov8n.pt")   # small model, decent FPS on CPU


# Experience buffer setup (saved across runs)
experience_file = Path("saved_experience") / "experience.pkl"
experience_file.parent.mkdir(parents=True, exist_ok=True)

experience = PILCOExperienceStorage(
    max_size=10000,             # limit stored transitions
    min_size_for_training=50,   # don’t train until we have enough data
    storage_file=str(experience_file),
    removal_strategy='fifo'     # drop oldest transitions first
)

pilco_policy = None
pilco_model = None  # used only during training

prev_state = None
prev_action = None


# Try restoring an old trained policy if one exists
print("\n[POLICY] Attempting to load saved policy...")
loaded_pilco, loaded_controller, experience_metadata = load_policy()

if loaded_pilco is not None:
    pilco_model = loaded_pilco
    pilco_policy = PILCOControllerWrapper(loaded_controller)
    print(f"[POLICY] Loaded policy with {len(experience)} transitions")
else:
    print("[POLICY] No saved policy found. Train first with exploration.")


# Each flight gets tagged so experience is grouped by session
experience.start_new_flight()

# State machine for what the drone should be doing
current_phase = 'wait'
test_step_count = 0
training_in_progress = False


print("\nControls:")
print("  m - Takeoff")
print("  e - Exploration (random data collection)")
print("  r - Train PILCO on gathered data")
print("  t - Test trained policy")
print("  l - Land")
print("  q - Quit + save")
print(f"Experience: {len(experience)} transitions from {len(set(experience.flight_ids))} flights")


# Small helper for consistent takeoff behavior
def safe_takeoff(drone):
    try:
        # battery read sometimes fails, so wrap it
        try:
            batt = drone.get_battery()
            print(f"[TAKEOFF] Battery: {batt}%")
            if batt < 15:
                print("[TAKEOFF] Battery too low.")
                return False
        except Exception as e:
            print(f"[TAKEOFF] Couldn't read battery: {e}")

        # reset RC before takeoff just in case a stale command is stuck
        try:
            drone.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            print(f"[TAKEOFF] RC reset failed: {e}")

        time.sleep(0.1)

        print("[TAKEOFF] Attempting takeoff...")
        drone.takeoff()
        time.sleep(1.2)  # give it a moment to stabilize

        print("[TAKEOFF] Success.")
        return True

    except Exception as e:
        print(f"[TAKEOFF] Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False


while True:
    # Grab frame from the drone’s camera
    frame = telloGetFrame(myDrone, w, h)

    # YOLO detection (flower info packaged by your helper)
    frame, info = detectFlower(frame, model)
    (cx, cy), area, conf, flower_score = info

    # Convert detection → 1D state for PILCO (your get_state wrapper)
    state = get_state(cx, cy, area, w, h)

    # Pick action depending on the current mode
    if current_phase == 'explore':
        # random yaw to generate variety for PILCO
        action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

    elif current_phase == 'test':
        # use trained policy if available
        if pilco_policy is not None:
            action = pilco_policy.compute_action(state)
        else:
            # fallback if user tries to test with no model
            action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

    else:
        # waiting: drone shouldn't rotate
        action = np.array([0.0], dtype=np.float32)

    # Convert normalized yaw → scaled RC command
    raw_yaw = action[0] * yaw_scale

    # small deadzone to avoid micro jitter
    if abs(raw_yaw) < 20:
        yaw_cmd = 0
    else:
        yaw_cmd = int(raw_yaw)

    # Default translational movement
    left_cmd = 0
    forward_cmd = 0

    try:
        if current_phase == 'explore':
            # explore uses only lateral jitter, no forward motion
            left_cmd = int(np.random.randint(-explore_lr_speed_max, explore_lr_speed_max + 1))
            forward_cmd = 0

        elif current_phase == 'test':
            # only move toward the flower if confident enough
            try:
                if conf is not None and conf >= flower_confidence_threshold:
                    forward_cmd = int(test_forward_speed)
            except Exception:
                forward_cmd = 0

        # send command to drone each loop
        myDrone.send_rc_control(left_cmd, forward_cmd, 0, yaw_cmd)

    except Exception as e:
        print(f"[ERROR] RC control failed: {e}")

    # Save transition (previous state/action → new state)
    if prev_state is not None:
        experience.add(prev_state, prev_action, state)

    prev_state = state
    prev_action = action
    step_count += 1

    # HUD overlays for debugging
    cv2.putText(frame, f"yaw: {yaw_cmd}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Pick overlay color based on active phase
    if training_in_progress:
        phase_text = "Phase: TRAINING"
        phase_color = (0, 140, 255)
    else:
        phase_text = f"Phase: {current_phase.upper()}"
        phase_color = (0, 255, 0) if current_phase == 'explore' else \
                      (255, 0, 0) if current_phase == 'test' else (200, 200, 0)

    cv2.putText(frame, phase_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)

    # small stats overlay
    exp_stats = experience.get_stats()
    cv2.putText(frame, f"Transitions: {exp_stats['total_transitions']} | Flights: {exp_stats['num_flights']}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # right side control cheatsheet
    controls = [
        "Controls:",
        "m: Takeoff",
        "l: Land",
        "e: Exploration",
        "r: Train PILCO",
        "t: Test policy",
        "q: Quit"
    ]

    cx_text = w - 10
    cy = 20
    for line in controls:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, line, (cx_text - tw, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1)
        cy += int(th * 1.6)

    cv2.imshow("Autonomous Bee Drone (PILCO)", frame)

    # handle keyboard
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n[POLICY] Saving before exit...")
        experience.end_flight()

        if pilco_model is not None and pilco_policy is not None:
            controller = pilco_policy.controller if hasattr(pilco_policy, "controller") else None
            if controller is not None:
                save_policy(pilco_model, controller, experience)

        try:
            myDrone.land()
        except Exception as e:
            print(f"[ERROR] Land failed: {e}")

        break

    elif key == ord('m'):
        safe_takeoff(myDrone)
        experience.start_new_flight()
        step_count = 0
        current_phase = 'wait'
        print("[TAKEOFF] Drone ready. Use 'e' to explore or 't' to test.")

    elif key == ord('l'):
        try:
            myDrone.land()
            experience.end_flight()
            print("[LAND] Landed.")
        except Exception as e:
            print(f"[ERROR] Land failed: {e}")

    elif key == ord('e'):
        current_phase = 'explore'
        step_count = 0
        print("[EXPLORE] Collecting random data...")

    elif key == ord('r'):
        training_in_progress = True
        current_phase = 'training'

        # quick visual feedback that training started
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
            print("\n[POLICY] Training PILCO...")

            try:
                S, A, S2 = experience.get()
                print(f"[POLICY] Training with {len(S)} transitions from {len(set(experience.flight_ids))} flights")

                pilco, controller = train_pilco(S, A, S2)
                pilco_model = pilco
                pilco_policy = PILCOControllerWrapper(controller)

                save_policy(pilco_model, controller, experience)
                print("[POLICY] Policy saved.")

                # free GP memory (PILCO models can be huge)
                try:
                    if hasattr(pilco_model, "gps"):
                        pilco_model.gps = None
                    pilco_model = None
                    gc.collect()
                except Exception as e:
                    print(f"[POLICY] Couldn't free memory: {e}")

                current_phase = 'wait'
                print("[POLICY] Training done. Use 't' to test or 'e' to gather more data.")

            finally:
                training_in_progress = False

        else:
            print(f"[POLICY] Not enough data. Need {experience.min_size_for_training}, have {len(experience)}.")

            try:
                overlay_frame = frame.copy()
                cv2.putText(overlay_frame, "Phase: TRAINING", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                cv2.putText(overlay_frame, "Not enough data", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 1)
                cv2.imshow("Autonomous Bee Drone (PILCO)", overlay_frame)
                cv2.waitKey(500)
            except Exception:
                pass

            training_in_progress = False
            current_phase = 'wait'

    elif key == ord('t'):
        if pilco_policy is not None:
            current_phase = 'test'
            test_step_count = 0
            print("[TEST] Running trained policy...")
        else:
            print("[TEST] No policy yet. Train first with 'r'.")
