import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from perception_yolo import YoloPerception, discretize_bbox_state
from qlearning import QLearnConfig, QLearningAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, choices=["2d", "pybullet"], default="2d")
    parser.add_argument("--model", type=str, default=str((_THIS_DIR / ".." / "AutonomousBeeDrone" / "yolov8n.pt").resolve()))
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--q", type=str, default=str((_THIS_DIR / "artifacts" / "q_table.npy").resolve()))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    target_class = args.target_class

    if args.sim == "pybullet":
        gui = True
        if args.no_gui:
            gui = False
        elif args.gui:
            gui = True

        from sim_env import PyBulletFollowSim, SimConfig

        cfg = SimConfig(gui=gui)
        sim = PyBulletFollowSim(cfg)
        if target_class is None:
            target_class = "sports ball"
    else:
        from sim_env_2d import Kinematic2DFollowSim, Sim2DConfig

        cfg = Sim2DConfig()
        sim = Kinematic2DFollowSim(cfg)
        if target_class is None:
            target_class = "bus"

    sim.reset()

    detector = YoloPerception(model_path=args.model, target_class=target_class, conf_thresh=0.25, device="cpu", imgsz=args.imgsz)

    agent = QLearningAgent(n_states=30, n_actions=5, cfg=QLearnConfig(eps_start=0.0))
    agent.load(args.q)
    agent.eps = 0.0

    for t in range(args.steps):
        frame_rgb = sim.get_camera_frame_rgb()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        det = detector.detect(frame_bgr)

        if det.ok and det.xyxy is not None:
            s, _, _, _ = discretize_bbox_state(det.xyxy, *sim.img_size)
            det_xyxy = det.xyxy
        else:
            s = 0
            det_xyxy = None

        a = agent.act(s)
        sim.step(a)

        if args.display:
            out = frame_bgr.copy()
            if det_xyxy is not None:
                x1, y1, x2, y2 = [int(v) for v in det_xyxy]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"t {t+1}/{args.steps} a {a}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Sim2 Q-Learn Follow (run)", out)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    sim.close()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
