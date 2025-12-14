import argparse
import math
from pathlib import Path
import sys
from typing import Optional, Tuple

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, choices=["pybullet"], default="pybullet")
    parser.add_argument("--model", type=str, default=_default_model_path())
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--q", type=str, default=str((_THIS_DIR / "artifacts" / "q_table.npy").resolve()))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--out-video", type=str, default=None)
    parser.add_argument("--out-fps", type=float, default=30.0)
    parser.add_argument("--search-when-lost", action="store_true")
    parser.add_argument("--print-every", type=int, default=30)
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

    sim.reset()

    detector = YoloPerception(model_path=args.model, target_class=target_class, conf_thresh=0.25, device="cpu", imgsz=args.imgsz)

    agent = QLearningAgent(n_states=30, n_actions=5, cfg=QLearnConfig(eps_start=0.0))
    agent.load(args.q)
    agent.eps = 0.0

    w, h = sim.img_size

    writer = None
    if args.out_video is not None:
        out_path = Path(args.out_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(args.out_fps), (w, h))
        if not writer.isOpened():
            writer = None

    had_det_count = 0
    centered_count = 0
    closeness_sum = 0.0
    lost_streak = 0
    search_dir = 1

    def _draw_debug(frame_bgr: np.ndarray, det_xyxy: Optional[Tuple[float, float, float, float]], text: str) -> np.ndarray:
        out = frame_bgr.copy()
        x1 = int(w / 3)
        x2 = int(2 * w / 3)
        cv2.line(out, (x1, 0), (x1, h - 1), (255, 255, 255), 1)
        cv2.line(out, (x2, 0), (x2, h - 1), (255, 255, 255), 1)

        # Camera center crosshair
        cx = int(w / 2)
        cy = int(h / 2)
        cv2.drawMarker(out, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

        if det_xyxy is not None:
            bx1, by1, bx2, by2 = [int(v) for v in det_xyxy]
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return out

    for t in range(args.steps):
        frame_rgb = sim.get_camera_frame_rgb()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        det = detector.detect(frame_bgr)

        cx_norm = -1.0

        if det.ok and det.xyxy is not None:
            s, region, _, close = discretize_bbox_state(det.xyxy, *sim.img_size)
            det_xyxy = det.xyxy
            cx_norm = float((det.xyxy[0] + det.xyxy[2]) / 2.0) / float(w)
            had_det_count += 1
            lost_streak = 0
            if region == 1:
                centered_count += 1
            closeness_sum += float(close)
        else:
            s = 0
            det_xyxy = None
            region = -1
            close = 0.0
            lost_streak += 1

        if det_xyxy is not None:
            search_dir = -1 if region == 0 else (1 if region == 2 else search_dir)

        if args.search_when_lost and lost_streak >= 8:
            # Simple reacquisition policy:
            # - sweep yaw for a while
            # - occasionally move forward to change viewpoint
            if (lost_streak % 30) == 0:
                search_dir *= -1
            if (lost_streak % 10) == 0:
                a = 3
            else:
                a = 1 if search_dir < 0 else 2
        else:
            a = agent.act(s)
        sim.step(a)

        # True centering check: within +/- 5% of image width.
        cx_err = float(cx_norm) - 0.5 if cx_norm >= 0.0 else 0.0
        centered = bool(cx_norm >= 0.0 and abs(cx_err) <= 0.05)
        debug_txt = (
            f"t {t+1}/{args.steps} a {a} det {int(det_xyxy is not None)} "
            f"cx {cx_norm:.3f} err {cx_err:+.3f} cen {int(centered)} reg {region} close {close:.3f} lost {lost_streak}"
        )
        if args.print_every > 0 and ((t + 1) % int(args.print_every) == 0):
            print(debug_txt)
        if args.display or writer is not None:
            out = _draw_debug(frame_bgr, det_xyxy, debug_txt)
            if writer is not None:
                writer.write(out)
            if args.display:
                cv2.imshow("Sim2 Q-Learn Follow (run)", out)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    det_rate = float(had_det_count) / float(max(1, args.steps))
    centered_rate = float(centered_count) / float(max(1, had_det_count))
    avg_close = float(closeness_sum) / float(max(1, had_det_count))
    print(f"steps={args.steps} det_rate={det_rate:.3f} centered_rate={centered_rate:.3f} avg_closeness={avg_close:.4f}")

    sim.close()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
