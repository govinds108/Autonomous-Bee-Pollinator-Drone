# Autonomous Bee Pollinator Drone

**Purpose**: Autonomous Bee Pollinator Drone is an experimental project that uses a small drone (Tello) with a YOLO-based visual detector and a PILCO-based controller to practice autonomous flower tracking and yaw control. It collects flight experience, trains a simple PILCO model (using RBF controller + custom GPs), and then tests the learned policy.

**Repository structure (important files)**

- `AutonomousBeeDrone/AutonomousBeeDrone.py` — main application (camera loop, flight control, exploration→train→test cycle).
- `pilco_experience_storage.py` — persistent experience buffer for (s, a, s') transitions (`saved_experience/experience.pkl`).
- `pilco_training.py` — helper that prepares data and trains a SimplePILCO controller.
- `simple_pilco.py`, `simple_gp.py`, `rbf_controller.py` — toy PILCO/GP/controller implementations.
- `policy_persistence.py` — save/load policy metadata in `saved_policies/latest_policy.pkl`.
- `weights/` — model weights (excluded from git).
- `saved_experience/` and `saved_policies/` — persistent data (excluded from git).
- `yolov8n.pt` — YOLOv8-nano weights (must be present in `AutonomousBeeDrone/`)

**Quickstart (macOS / zsh)**

1. Create and activate the virtualenv (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Download YOLO weights (if not present locally):

```bash
cd "$(pwd)"/AutonomousBeeDrone
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This will download `yolov8n.pt` into the current folder. If download fails, manually fetch:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

4. Run the main application:

```bash
cd AutonomousBeeDrone
python3 AutonomousBeeDrone.py
```

**Keyboard controls (in-app)**

- `t` — Takeoff (starts exploration)
- `l` — Land
- `n` — Skip TEST phase, return to EXPLORE
- `c` — Toggle consecutive cycles (auto cycle on/off)
- `r` — Manual retrain PILCO with current experience
- `s` — Save current policy manually
- `i` — Show experience statistics
- `q` — Quit and save

**Where flight metrics and models are stored**

- Experience buffer file: `AutonomousBeeDrone/saved_experience/experience.pkl` — contains the recorded (s, a, s') transitions.
- Saved policy: `AutonomousBeeDrone/saved_policies/latest_policy.pkl` — contains controller params and GP training data (no large precomputed `K`/`K_inv`).
- Model weights: `weights/` or `yolov8n.pt` in `AutonomousBeeDrone/`.

To clear or prune flight data safely, use the `PILCOExperienceStorage` API (see `pilco_experience_storage.py`) or create a small helper script to remove specific flight IDs. Back up files before destructive actions.
