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

explore_steps = 400
step_count = 0

yaw_scale = 40
test_forward_speed = 15

explore_forward_speed_max = 30
explore_lr_speed_max = 30

# INIT DRONE + YOLO MODEL
myDrone = initializeTello()

print("Loading YOLO-Nano model...")
model = YOLO("yolov8n.pt")

# EXPERIENCE STORAGE & POLICY
experience_file = Path("saved_experience") / "experience.pkl"
experience_file.parent.mkdir(parents=True, exist_ok=True)

experience = PILCOExperienceStorage(
    max_size=50000,
    min_size_for_training=50,
    storage_file=str(experience_file),
    removal_strategy='fifo'
)

pilco_policy = None
pilco_model = None

prev_state = None
prev_action = None

# Try to load previously saved policy
print("\n[POLICY] Attempting to load saved policy...")
loaded_pilco, loaded_controller, experience_metadata = load_policy()

if loaded_pilco is not None:
    pilco_model = loaded_pilco
    pilco_policy = PILCOControllerWrapper(loaded_controller)
    print(f"[POLICY] Loaded policy with {len(experience)} transitions")
else:
    print("[POLICY] No saved policy found.")

experience.start_new_flight()

# Phases: 'explore', 'test', 'wait'
current_phase = 'wait'
test_step_count = 0
training_in_progress = False

print("\n" + "="*60)
print("CONTROLS:")
print("  'm' - Takeoff")
print("  'e' - Start EXPLORATION")
print("  'r' - Save experience for offline training")
print("  't' - Start TESTING")
print("  'l' - Land")
print("  'q' - Quit and save")
print("="*60)
print(f"Experience: {len(experience)} transitions from {len(set(experience.flight_ids))} flights")

def safe_takeoff(drone):
    try:
        try:
            batt = drone.get_battery()
            print(f"[TAKEOFF] Battery: {batt}%")
            if batt < 15:
                print("[TAKEOFF] ERROR: Battery too low (<15%).")
                return False
        except Exception as e:
            print(f"[TAKEOFF] WARNING: Could not check battery: {e}")

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


# MAIN LOOP
while True:

    frame = telloGetFrame(myDrone, w, h)

    frame, info = detectFlower(frame, model)
    (cx, cy), area, conf, flower_score = info

    state = get_state(cx, cy, area, w, h)

    # ACTION SELECTION
    if current_phase == 'explore':
        yaw_rand = np.random.uniform(-1.0, 1.0)
        forward_rand = np.random.uniform(-0.2, 0.6)
        action = np.array([yaw_rand, forward_rand], dtype=np.float32)

    elif current_phase == 'test':
        if pilco_policy is not None:
            action = pilco_policy.compute_action(state)
        else:
            action = np.array([np.random.uniform(-1.0, 1.0), 0.0], dtype=np.float32)
    else:
        action = np.array([0.0, 0.0], dtype=np.float32)

    # Convert action to yaw velocity
    raw_yaw = float(action[0]) * yaw_scale

    if abs(raw_yaw) < 20:
        yaw_cmd = 0
    else:
        yaw_cmd = int(raw_yaw)

    try:
        left_cmd = 0
        forward_cmd = 0

        if current_phase == 'explore':
            left_cmd = int(np.random.randint(-explore_lr_speed_max, explore_lr_speed_max + 1))
            forward_cmd = int(action[1] * explore_forward_speed_max)

        elif current_phase == 'test':
            forward_cmd = int(np.clip(float(action[1]), -1.0, 1.0) * test_forward_speed)

        myDrone.send_rc_control(left_cmd, forward_cmd, 0, yaw_cmd)
    except Exception as e:
        print(f"[ERROR] RC control failed: {e}")

    # Store transition
    if prev_state is not None and current_phase in ['explore', 'test']:
        experience.add(prev_state, prev_action, state)

    prev_state = state
    prev_action = action
    step_count += 1

    # OVERLAYS
    cv2.putText(frame, f"yaw: {yaw_cmd}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    if training_in_progress:
        phase_text = "Phase: TRAINING"
        phase_color = (0, 140, 255)
    else:
        phase_text = f"Phase: {current_phase.upper()}"
        phase_color = (0, 255, 0) if current_phase == 'explore' else \
                      (255, 0, 0) if current_phase == 'test' else (200, 200, 0)

    cv2.putText(frame, phase_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)

    exp_stats = experience.get_stats()
    cv2.putText(frame, f"Transitions: {exp_stats['total_transitions']} | Flights: {exp_stats['num_flights']}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    controls = [
        "Controls:",
        "m: Takeoff",
        "l: Land",
        "e: Exploration",
        "t: Test",
        "q: Quit"
    ]

    cx_text = w - 10
    cy = 20
    for line in controls:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, line, (cx_text - tw, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        cy += int(th * 1.6)

    cv2.imshow("Autonomous Bee Drone (PILCO)", frame)

    # KEYBOARD INPUT
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n[POLICY] Saving before exit...")
        experience.end_flight()
        if pilco_model is not None and pilco_policy is not None:
            controller = pilco_policy.controller if hasattr(pilco_policy, 'controller') else None
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
        print("[TAKEOFF] SUCCESS. In WAIT state.")

    elif key == ord('l'):
        try:
            myDrone.land()
            experience.end_flight()
            print("[LAND] Landed successfully.")
        except Exception as e:
            print(f"[ERROR] Land failed: {e}")

    elif key == ord('e'):
        current_phase = 'explore'
        step_count = 0
        print("[EXPLORE] Starting exploration...")

    elif key == ord('r'):
        try:
            experience.end_flight()
            print("[POLICY] Saved current experience to disk.")
        except Exception as e:
            print(f"[POLICY] Warning: could not save experience: {e}")

        print("[POLICY] On-drone training is disabled. Use offline script.")
        training_in_progress = False
        current_phase = 'wait'

    elif key == ord('t'):
        if pilco_policy is not None:
            current_phase = 'test'
            test_step_count = 0
            print("[TEST] Starting testing...")
        else:
            print("[TEST] No trained policy available.")
