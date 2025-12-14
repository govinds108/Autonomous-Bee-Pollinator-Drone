# Sim2 (PyBullet + YOLO + Q-Learning)

This folder contains a simple 3D drone-following simulation (PyBullet) with an off-the-shelf YOLO detector (Ultralytics) and a Q-learning policy.

## What the demo does

- **Simulator:** PyBullet 3D scene (`Sim2/sim_env.py`) with a drone body and a target ball.
- **Perception:** YOLO runs on the simulator camera frame (`Sim2/perception_yolo.py`).
- **State (30 total):** left/center/right (3) × closeness bin (10). Closeness is `bbox_area / image_area`.
- **Action space (5):** noop, yaw-left, yaw-right, forward, backward.
- **Training:** Q-learning (`Sim2/train_qlearn_follow.py`) produces a Q-table `Sim2/artifacts/q_table.npy`.
- **Run:** `Sim2/run_qlearn_follow.py` loads the Q-table and executes the learned policy.

## Docker

### Build

From the repo root:

```bash
docker build -t sim2 -f Sim2/Dockerfile .
```

### Run training (headless)

The default `CMD` in the Dockerfile runs training headless. You can run it explicitly:

```bash
docker run --rm sim2 python Sim2/train_qlearn_follow.py --no-gui --episodes 80 --steps 200 --rand-start-y 0.6 --rand-start-yaw-deg 45
```

This writes `Sim2/artifacts/q_table.npy` inside the container.

### Run the learned policy (headless + printed metrics)

To see whether the policy is centering the target and moving toward it, use the printed debug/summary metrics:

```bash
docker run --rm sim2 python Sim2/run_qlearn_follow.py --no-gui --q Sim2/artifacts/q_table.npy --steps 600 --search-when-lost --print-every 30
```

At the end it prints:

- `det_rate`: fraction of frames YOLO detected the target
- `centered_rate`: fraction of detected frames where the target is centered
- `avg_closeness`: average `bbox_area / image_area` when detected

Example periodic debug line:

```
t 30/600 a 0 det 1 cx 0.494 err -0.006 cen 1 reg 1 close 0.595 lost 0
```

Where:

- `cx` is bbox center-x normalized to [0..1]
- `err = cx - 0.5` (0 means perfect camera-center)
- `cen` is 1 if `abs(err) <= 0.05`

### GUI (optional)

The Dockerfile is configured for headless runs. Showing PyBullet GUI and OpenCV windows from Docker requires extra host setup (X11/Wayland forwarding, permissions, etc.).

### Train

```bash
python Sim2/train_qlearn_follow.py --no-gui --episodes 80 --steps 200 --rand-start-y 0.6 --rand-start-yaw-deg 45
```

### Run (GUI)

```bash
python Sim2/run_qlearn_follow.py --gui --q Sim2/artifacts/q_table.npy --start-y 0.4 --start-yaw-deg 20 --steps 600 --search-when-lost --print-every 30
```

### Run with camera window + bbox overlay

On some Ubuntu Wayland setups, OpenCV `--display` requires forcing Qt to X11:

```bash
QT_QPA_PLATFORM=xcb python Sim2/run_qlearn_follow.py --gui --display --q Sim2/artifacts/q_table.npy --start-y 0.4 --start-yaw-deg 20 --steps 600 --search-when-lost
```
