import time, math, json, subprocess
import numpy as np
from PIL import Image
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


# --- config: point this to your conda yolo python ---
YOLO_PY = "/opt/anaconda3/envs/yolo/bin/python"
YOLO_SCRIPT = "yolo_bbox_only.py"
STEP_IMG = "step.png"

BALL_POS = (0.8, 0.0, 0.08)


def spawn_ball(pos=BALL_POS, radius=0.12):
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
    return p.createMultiBody(0, collision, visual, list(pos))


def get_frame(cam_target, width=640, height=480):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(cam_target),
        distance=1.2,
        yaw=35,
        pitch=-20,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(60, width/height, 0.05, 50.0)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    return rgba[:, :, :3]


def bbox_to_state(x1, y1, x2, y2, img_w, img_h, n_bins=10):
    cx = (x1 + x2) / 2.0
    if cx < img_w/3:
        region = 0
    elif cx < 2*img_w/3:
        region = 1
    else:
        region = 2

    area = max(0.0, x2-x1) * max(0.0, y2-y1)
    closeness = area / float(img_w * img_h)
    b = int(math.floor(closeness * n_bins))
    b = max(0, min(n_bins-1, b))
    return region*10 + b, region, b, closeness


def action_to_rpm(a, base=40000, dyaw=2500):
    # 0 hover, 1 yaw left, 2 yaw right
    rpm = np.array([base, base, base, base], dtype=np.float32)
    if a == 1:
        rpm += np.array([+dyaw, -dyaw, +dyaw, -dyaw], dtype=np.float32)
    elif a == 2:
        rpm += np.array([-dyaw, +dyaw, -dyaw, +dyaw], dtype=np.float32)
    return rpm.reshape(1,4)


def yolo_bbox_from_image(img_path=STEP_IMG):
    out = subprocess.check_output([YOLO_PY, YOLO_SCRIPT, img_path], text=True).strip()
    data = json.loads(out)
    return data


def reward_fn(region, closeness, prev_close, action):
    r = 0.0
    # keep centered
    r += 1.0 if region == 1 else -1.0
    # get closer
    if prev_close is not None:
        r += 5.0 * (closeness - prev_close)
    # discourage hovering forever
    if action == 0:
        r -= 0.2
    return r


def train():
    EPISODES = 30
    STEPS = 150
    ALPHA = 0.2
    GAMMA = 0.95
    EPS = 1.0
    EPS_DECAY = 0.97
    EPS_MIN = 0.1

    N_STATES = 30
    N_ACTIONS = 3  # hover, yawL, yawR
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)

    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, physics=Physics.PYB, gui=True)

    for ep in range(EPISODES):
        env.reset()

        # spawn ball every episode and aim GUI camera at it
        spawn_ball(BALL_POS, radius=0.12)
        p.resetDebugVisualizerCamera(1.2, 35, -20, list(BALL_POS))

        prev_close = None
        ep_ret = 0.0

        for t in range(STEPS):
            frame = get_frame(BALL_POS, 640, 480)
            Image.fromarray(frame).save(STEP_IMG)

            det = yolo_bbox_from_image(STEP_IMG)
            if not det.get("ok", False):
                # lost detection -> punish and take random recovery action
                r = -3.0
                a = np.random.randint(N_ACTIONS)
                env.step(action_to_rpm(a))
                time.sleep(1/60)
                ep_ret += r
                prev_close = None
                continue

            x1,y1,x2,y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            s, region, _, close = bbox_to_state(x1,y1,x2,y2, 640,480)

            # epsilon-greedy
            if np.random.rand() < EPS:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[s]))

            # act
            env.step(action_to_rpm(a))
            time.sleep(1/60)

            # next state
            frame2 = get_frame(BALL_POS, 640, 480)
            Image.fromarray(frame2).save(STEP_IMG)
            det2 = yolo_bbox_from_image(STEP_IMG)

            if det2.get("ok", False):
                s2, region2, _, close2 = bbox_to_state(det2["x1"], det2["y1"], det2["x2"], det2["y2"], 640,480)
                r = reward_fn(region2, close2, prev_close, a)
                Q[s, a] = (1-ALPHA)*Q[s,a] + ALPHA*(r + GAMMA*np.max(Q[s2]))
                prev_close = close2
            else:
                r = -3.0
                prev_close = None

            ep_ret += r

        EPS = max(EPS_MIN, EPS*EPS_DECAY)
        print(f"ep {ep+1}/{EPISODES} return={ep_ret:.2f} eps={EPS:.2f}")

    env.close()
    np.save("Q_yaw.npy", Q)
    print("Saved Q_yaw.npy")


if __name__ == "__main__":
    train()
