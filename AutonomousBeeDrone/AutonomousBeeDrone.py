from utils import *
from pilco_data import ReplayBufferPILCO
from pilco_training import train_pilco
from pilco_controller_wrapper import PILCOControllerWrapper

import cv2
import numpy as np
from ultralytics import YOLO
import time


# ===========================
# SETTINGS
# ===========================
w, h = 720, 440

explore_mode = True            # first flight collects data
explore_steps = 800            # ~8 seconds at ~100Hz
step_count = 0

yaw_scale = 40                 # smoother, safer yaw control


# ===========================
# INIT DRONE + YOLO MODEL
# ===========================
myDrone = initializeTello()

print("Loading YOLO-Nano model...")
model = YOLO("yolov8n.pt")

# Memory for PILCO
buffer = ReplayBufferPILCO()

pilco_policy = None

prev_state = None
prev_action = None

print("Press 't' to TAKEOFF, 'l' to land, 'q' to quit.")
print("Mode:", "Exploration" if explore_mode else "PILCO")


# ============================================================
# SAFE TAKEOFF FUNCTION (fixed)
# ============================================================
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
    # CHOOSE ACTION
    # -------------------------
    if explore_mode:
        # Random yaw exploration
        action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
    else:
        # PILCO policy
        action = pilco_policy.compute_action(state)  # [-1, 1]

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
    # Send command (only yaw)
    # -------------------------
    try:
        myDrone.send_rc_control(0, 0, 0, yaw_cmd)
    except Exception as e:
        print(f"[ERROR] RC control failed: {e}")

    # -------------------------
    # Store transition (s, a, s')
    # -------------------------
    if prev_state is not None:
        buffer.add(prev_state, prev_action, state)

    prev_state = state
    prev_action = action
    step_count += 1

    # -------------------------
    # TRAIN PILCO AFTER EXPLORATION
    # -------------------------
    if explore_mode and step_count >= explore_steps:
        print("Exploration complete. Training PILCO model...")

        S, A, S2 = buffer.get()
        pilco, controller = train_pilco(S, A, S2)
        pilco_policy = PILCOControllerWrapper(controller)

        explore_mode = False
        print("PILCO training complete. Switching to control mode.")

    # -------------------------
    # OVERLAYS
    # -------------------------
    cv2.putText(frame, f"yaw: {yaw_cmd}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame,
                f"Mode: {'Explore' if explore_mode else 'PILCO'}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,0), 2)

    cv2.imshow("Autonomous Bee Drone (PILCO)", frame)

    # -------------------------
    # KEYBOARD INPUT
    # -------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        try: myDrone.land()
        except Exception as e: print(f"[ERROR] Land failed: {e}")
        break

    elif key == ord('t'):
        safe_takeoff(myDrone)

    elif key == ord('l'):
        try: myDrone.land()
        except Exception as e: print(f"[ERROR] Land failed: {e}")
