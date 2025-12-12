# Autonomous Bee Pollinator Drone

An experimental project using a Tello drone with YOLO detection and PILCO control for autonomous flower tracking and yaw control. Collects flight data, trains a PILCO model, and tests the policy.

## Repository Structure

- `AutonomousBeeDrone/AutonomousBeeDrone.py` — Main application with camera loop and flight control.
- `pilco_experience_storage.py` — Experience buffer for transitions.
- `pilco_training.py` — Trains PILCO model.
- `simple_pilco.py`, `simple_gp.py`, `rbf_controller.py` — PILCO/GP implementations.
- `policy_persistence.py` — Saves/loads policies.
- `weights/` — Model weights.
- `saved_experience/` and `saved_policies/` — Persistent data.
- `yolov8n.pt` — YOLOv8-nano weights.

## Quickstart

1. Create and activate virtualenv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLO weights:

   ```bash
   cd AutonomousBeeDrone
   python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

   Or download manually from: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

4. Run:
   ```bash
   cd AutonomousBeeDrone
   python3 AutonomousBeeDrone.py
   ```

## Controls

- `t` — Takeoff
- `l` — Land
- `n` — Skip test, back to explore
- `c` — Toggle auto cycles
- `r` — Retrain PILCO
- `s` — Save policy
- `i` — Show stats
- `q` — Quit and save

## Storage

- Experience: `AutonomousBeeDrone/saved_experience/experience.pkl`
- Policies: `AutonomousBeeDrone/saved_policies/latest_policy.pkl`
- Weights: `weights/` or `yolov8n.pt`
