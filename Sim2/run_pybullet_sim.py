import argparse
import time

import numpy as np

from sim_env import PyBulletFollowSim, SimConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=None, help="Optional wall-clock sleep per step when running headless")
    args = parser.parse_args()

    gui = True
    if args.no_gui:
        gui = False
    elif args.gui:
        gui = True

    sim = PyBulletFollowSim(SimConfig(gui=gui))
    sim.reset()

    # Simple demo policy: alternate yaw and forward/back to show motion.
    actions = [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4]

    for t in range(args.steps):
        a = actions[t % len(actions)]
        sim.step(a)

        # Grab a frame so we know the camera pipeline is working.
        frame_rgb = sim.get_camera_frame_rgb()
        _ = np.asarray(frame_rgb)

        if (t + 1) % 120 == 0:
            print(f"t={t+1}/{args.steps} action={a}")

        if args.sleep is not None and not gui:
            time.sleep(float(args.sleep))

    sim.close()


if __name__ == "__main__":
    main()
