from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import numpy as np
import time

# Create the environment
env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    gui=True,              # IMPORTANT: shows the simulator window
    record=False
)

obs, info = env.reset()
print("Environment reset successful")

# Run for a few seconds
for i in range(300):
    # Hover action (RPMs for 4 motors)
    action = np.array([[40000, 40000, 40000, 40000]])
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/60)

env.close()
