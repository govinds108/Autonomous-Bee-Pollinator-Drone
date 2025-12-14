import time
import numpy as np
from PIL import Image

import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def get_pybullet_frame(width=640, height=480):
    """Render an RGB frame from the current PyBullet scene."""
    # Camera looking toward the origin (0,0,0) from a point behind/above it
    cam_target = [0, 0, 0.5]
    cam_distance = 2.0
    cam_yaw = 45
    cam_pitch = -30
    cam_roll = 0

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
        distance=cam_distance,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=cam_roll,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.05,
        farVal=50.0
    )

    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    rgb = rgba[:, :, :3]
    return rgb


# 1) Start simulator (this opens PyBullet + loads drone)
env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    gui=True
)

obs, info = env.reset()
print("Simulator running")

# 2) Step a bit so things settle
for _ in range(60):
    action = np.array([[40000, 40000, 40000, 40000]])
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/60)

# 3) Capture a frame directly from PyBullet
frame = get_pybullet_frame(width=640, height=480)
print("Frame shape:", frame.shape, "dtype:", frame.dtype)

# 4) Save it
Image.fromarray(frame).save("frame0.png")
print("Saved frame0.png")

# 5) Close sim
time.sleep(60)  # keep GUI open for 10 seconds
env.close()
