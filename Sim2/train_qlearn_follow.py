import argparse
import math
from pathlib import Path
from typing import Optional, Tuple
import sys

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from perception_yolo import YoloPerception, discretize_bbox_state
from qlearning import QLearnConfig, QLearningAgent


def _default_model_path() -> str:
    p = (_THIS_DIR / ".." / "AutonomousBeeDrone" / "yolov8n.pt").resolve()
    if p.exists():
        return str(p)
    return "yolov8n.pt"


A_NOOP = 0
A_YAW_LEFT = 1
A_YAW_RIGHT = 2
A_FORWARD = 3
A_BACKWARD = 4


def reward_fn(
    region: int,
    closeness: float,
    prev_close: Optional[float],
    action: int,
    had_detection: bool,
) -> float:
    if not had_detection:
        return -3.0

    r = 0.0

    r += 1.0 if region == 1 else -1.0

    if prev_close is not None:
        r += 5.0 * (float(closeness) - float(prev_close))

    if action == A_NOOP:
        r -= 0.6

    if region == 1 and action == A_FORWARD:
        r += 0.4

    if region == 1 and (action == A_YAW_LEFT or action == A_YAW_RIGHT):
        r -= 0.1

    if region == 0 and action == A_YAW_RIGHT:
        r -= 0.5
    if region == 2 and action == A_YAW_LEFT:
        r -= 0.5

    return float(r)


def annotate(frame_bgr: np.ndarray, det_xyxy: Optional[Tuple[float, float, float, float]], text: str) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (int(w / 2), int(h / 2)), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
    if det_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in det_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, choices=["pybullet"], default="pybullet")
    parser.add_argument("--model", type=str, default=_default_model_path())
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--rand-start-y", type=float, default=0.0)
    parser.add_argument("--rand-start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=str((_THIS_DIR / "artifacts" / "q_table.npy").resolve()))
    args = parser.parse_args()

    target_class = args.target_class

    gui = True
    if args.no_gui:
        gui = False
    elif args.gui:
        gui = True

    from sim_env import PyBulletFollowSim, SimConfig

    cfg = SimConfig(
        gui=gui,
        drone_start_pos=(0.0, float(args.start_y), 0.4),
        drone_start_yaw=math.radians(float(args.start_yaw_deg)),
    )
    sim = PyBulletFollowSim(cfg)
    if target_class is None:
        target_class = "sports ball"

    base_start_pos = (0.0, float(args.start_y), 0.4)
    base_start_yaw = math.radians(float(args.start_yaw_deg))

    detector = YoloPerception(model_path=args.model, target_class=target_class, conf_thresh=0.25, device="cpu", imgsz=args.imgsz)

    agent = QLearningAgent(n_states=30, n_actions=5, cfg=QLearnConfig())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        if args.seed is not None:
            np.random.seed(int(args.seed) + int(ep))

        dy = float(args.rand_start_y) * (2.0 * float(np.random.rand()) - 1.0)
        dyaw = math.radians(float(args.rand_start_yaw_deg)) * (2.0 * float(np.random.rand()) - 1.0)
        start_pos = (base_start_pos[0], base_start_pos[1] + dy, base_start_pos[2])
        start_yaw = float(base_start_yaw) + float(dyaw)

        sim.reset(
            seed=None if args.seed is None else int(args.seed) + int(ep),
            drone_start_pos=start_pos,
            drone_start_yaw=start_yaw,
        )

        prev_close = None
        ep_ret = 0.0

        frame_rgb = sim.get_camera_frame_rgb()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        det = detector.detect(frame_bgr)

        if det.ok and det.xyxy is not None:
            s, region, _, close = discretize_bbox_state(det.xyxy, *sim.img_size)
            prev_close = close
        else:
            s, region, close = 0, 1, 0.0
            prev_close = None

        for t in range(args.steps):
            a = agent.act(s)
            sim.step(a)

            frame_rgb2 = sim.get_camera_frame_rgb()
            frame_bgr2 = cv2.cvtColor(frame_rgb2, cv2.COLOR_RGB2BGR)
            det2 = detector.detect(frame_bgr2)

            had_det = bool(det2.ok and det2.xyxy is not None)

            if had_det:
                s2, region2, _, close2 = discretize_bbox_state(det2.xyxy, *sim.img_size)
                r = reward_fn(region2, close2, prev_close, a, had_det)
                prev_close = close2
                det_xyxy = det2.xyxy
            else:
                s2, region2, close2 = 0, 1, 0.0
                r = reward_fn(region2, close2, prev_close, a, had_det)
                prev_close = None
                det_xyxy = None

            agent.update(s, a, r, s2)
            s = s2
            ep_ret += float(r)

            if args.display:
                txt = f"ep {ep+1}/{args.episodes} t {t+1}/{args.steps} r {r:.2f} eps {agent.eps:.2f}"
                vis = annotate(frame_bgr2, det_xyxy, txt)
                cv2.imshow("Sim2 Q-Learn Follow", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        agent.end_episode()
        print(f"ep {ep+1}/{args.episodes} return={ep_ret:.2f} eps={agent.eps:.2f}")

    agent.save(str(out_path))
    print(f"Saved {out_path}")

    sim.close()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
