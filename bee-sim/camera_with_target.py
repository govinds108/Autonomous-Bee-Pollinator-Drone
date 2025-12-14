import time
import numpy as np
from PIL import Image

import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def add_ball():
    radius = 0.06
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[1, 0.5, 0, 1]  # orange ball
    )
    pos = [0.8, 0.0, 0.06]
    ball_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=pos
    )
    return ball_id, pos


def get_frame(cam_target, width=640, height=480):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
        distance=1.2,
        yaw=25,
        pitch=-15,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.05, farVal=50.0
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER
    )
    rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    return rgba[:, :, :3]


env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    gui=True
)
env.reset()
print("Simulator running")

ball_id, ball_pos = add_ball()
print("Ball placed at", ball_pos)

for _ in range(120):
    env.step(np.array([[40000, 40000, 40000, 40000]]))
    time.sleep(1/60)

frame = get_frame([ball_pos[0], ball_pos[1], ball_pos[2]])
Image.fromarray(frame).save("frame_ball.png")
print("Saved frame_ball.png")

input("Press Enter to close...")
env.close()
