from utils import *
from sac_agent import SACAgent, ReplayBuffer
import cv2
import numpy as np
from ultralytics import YOLO

w, h = 720, 440
myDrone = initializeTello()
device = "cpu"

# Initialize SAC
state_dim = 4       # [x_error, y_error, area_norm, prev_error]
action_dim = 3      # [yaw, up_down, forward]
agent = SACAgent(state_dim, action_dim, device=device)
replay_buffer = ReplayBuffer(max_size=50000)
pError = 0

print("Loading YOLO-Nano model...")
model = YOLO('yolov8n.pt')
print("YOLO model loaded.")
yolo_confidence = 0.25
flower_score_threshold = 0.4

# Inform user how to operate (manual takeoff)
print("Manual takeoff enabled. Press 't' to takeoff, 'l' to land, 'q' to quit.")

while True:
    # Get drone camera frame
    img = telloGetFrame(myDrone, w, h)

    # Detect flower using YOLO + visual filters (updated utils.detectFlower signature)
    img, info = detectFlower(img, model, conf_thresh=yolo_confidence,
                              flower_thresh=flower_score_threshold)
    (cx, cy), area = info[0], info[1]

    # Compute state features
    x_error = (cx - w / 2) / (w / 2)
    y_error = (cy - h / 2) / (h / 2)
    area_norm = area / (w * h)
    state = np.array([x_error, y_error, area_norm, pError])

    # Select action from SAC
    action = agent.select_action(state)
    yaw = int(np.clip(action[0] * 100, -100, 100))
    up_down = int(np.clip(action[1] * 50, -50, 50))
    forward = int(np.clip(action[2] * 30, -30, 30))

    # Send movement commands
    try:
        myDrone.send_rc_control(0, forward, up_down, yaw)
    except Exception:
        # ignore send failures (drone may be on ground / disconnected)
        pass

    # Compute reward
    reward = -abs(x_error) - abs(y_error) + 5 * area_norm
    done = 0 if area > 0 else 1

    # Next observation (same frame placeholder; could be from next iteration)
    next_state = np.array([x_error, y_error, area_norm, x_error])
    replay_buffer.push(state, action, reward, next_state, done)

    # Train SAC agent
    actor_loss, critic_loss = agent.train(replay_buffer, batch_size=128)
    if actor_loss is not None:
        print(f"Actor Loss: {actor_loss:.3f}, Critic Loss: {critic_loss:.3f}")

    # Update tracking error
    pError = x_error

    # Display feed
    cv2.imshow("Tello Flower Detection (SAC)", img)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        try:
            myDrone.land()
        except Exception:
            pass
        break
    elif key == ord('t'):
        # Manual takeoff with battery check and exception handling
        try:
            batt = None
            try:
                batt = myDrone.get_battery()
            except Exception:
                pass
            if batt is not None and batt < 20:
                print(f"Battery low ({batt}%). Aborting takeoff. Charge drone before flying.")
            else:
                try:
                    myDrone.takeoff()
                except Exception as e:
                    print("Takeoff failed:", e)
        except Exception:
            # catch-all to avoid loop exit on unexpected errors
            print("Takeoff command raised an error.")
    elif key == ord('l'):
        try:
            myDrone.land()
        except Exception as e:
            print("Land failed:", e)
    elif key == ord('c'):
        cv2.imwrite('captured_frame.jpg', img)
        print("Saved frame to captured_frame.jpg")
