"""
Offline training script for PILCO using saved experience.

Usage:
    python train_offline.py

This script will load the `saved_experience/experience.pkl` file (if present),
train PILCO using `pilco_training.train_pilco`, and save the resulting policy
in `saved_policies/latest_policy.pkl` using `policy_persistence.save_policy`.
"""

from pathlib import Path
from pilco_experience_storage import PILCOExperienceStorage
from pilco_training import train_pilco
from policy_persistence import save_policy


def main():
    exp_file = Path("saved_experience") / "experience.pkl"
    if not exp_file.exists():
        print(f"No experience file at {exp_file}. Run exploration flights first.")
        return

    exp = PILCOExperienceStorage(storage_file=str(exp_file))
    S, A, S2 = exp.get()
    if S.size == 0:
        print("No transitions found in experience. Collect exploration data first.")
        return

    print(f"Training PILCO with {len(S)} transitions from {len(set(exp.flight_ids))} flights...")
    pilco, controller = train_pilco(S, A, S2)

    save_policy(pilco, controller, exp)
    print("Offline training complete. Policy saved to saved_policies/latest_policy.pkl")


if __name__ == '__main__':
    main()
