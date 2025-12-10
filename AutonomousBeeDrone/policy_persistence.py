"""
Policy persistence utilities for PILCO.

Handles saving and restoring trained controllers and model parameters.
"""

import numpy as np
import pickle
import os
from pathlib import Path

POLICY_DIR = Path("saved_policies")
POLICY_DIR.mkdir(exist_ok=True)
DEFAULT_POLICY_FILE = POLICY_DIR / "latest_policy.pkl"


def save_policy(pilco, controller, experience_storage, policy_file=None):
    """
    Save controller parameters, PILCO model data, and basic metadata.
    """
    if policy_file is None:
        policy_file = DEFAULT_POLICY_FILE

    policy_data = {
        'controller': {
            'state_dim': controller.state_dim,
            'num_basis': controller.num_basis,
            'action_dim': getattr(controller, 'action_dim', 1),
            'centers': controller.centers,
            'sigma': controller.sigma,
            'weights': controller.weights
        },
        'pilco': {
            'X': pilco.X,
            'U': pilco.U,
            'Y': pilco.Y,
            'horizon': pilco.horizon,
            'lr': pilco.lr,
            'gps': []
        },
        'gp_hyperparams': [],
        'experience_metadata': {
            'total_transitions': len(experience_storage),
            'num_flights': len(set(experience_storage.flight_ids))
                if hasattr(experience_storage, 'flight_ids') else 0,
            'current_flight_id': getattr(experience_storage, 'current_flight_id', 0)
        }
    }

    # GP model parameters
    for gp in pilco.gps:
        gp_data = {
            'X': gp.X,
            'Y': gp.Y,
            'lengthscale': gp.lengthscale,
            'variance': gp.variance,
            'noise': gp.noise
        }
        policy_data['pilco']['gps'].append(gp_data)
        policy_data['gp_hyperparams'].append({
            'lengthscale': gp.lengthscale,
            'variance': gp.variance,
            'noise': gp.noise
        })

    with open(policy_file, 'wb') as f:
        pickle.dump(policy_data, f)

    print(f"[POLICY] Saved policy to {policy_file}")
    return policy_file


def load_policy(policy_file=None):
    """
    Load controller and PILCO model from disk.
    Returns (pilco, controller, metadata) or (None, None, None) if unavailable.
    """
    if policy_file is None:
        policy_file = DEFAULT_POLICY_FILE

    if not os.path.exists(policy_file):
        print(f"[POLICY] No saved policy found at {policy_file}")
        return None, None, None

    try:
        with open(policy_file, 'rb') as f:
            policy_data = pickle.load(f)

        from rbf_controller import RBFController
        controller_cfg = policy_data['controller']

        controller = RBFController(
            state_dim=controller_cfg['state_dim'],
            num_basis=controller_cfg['num_basis'],
            action_dim=controller_cfg.get('action_dim', 1)
        )
        controller.centers = controller_cfg['centers']
        controller.sigma = controller_cfg['sigma']
        controller.weights = controller_cfg['weights']

        from simple_pilco import SimplePILCO
        pilco_cfg = policy_data['pilco']

        pilco = SimplePILCO(
            X=pilco_cfg['X'],
            U=pilco_cfg['U'],
            Y=pilco_cfg['Y'],
            controller=controller,
            horizon=pilco_cfg['horizon'],
            lr=pilco_cfg['lr']
        )

        from simple_gp import SimpleGP
        pilco.gps = []

        for gp_data in pilco_cfg['gps']:
            gp = SimpleGP(
                X=gp_data['X'],
                Y=gp_data['Y'],
                lengthscale=gp_data['lengthscale'],
                variance=gp_data['variance'],
                noise=gp_data['noise']
            )
            gp._compute_kernel()
            pilco.gps.append(gp)

        metadata = policy_data.get('experience_metadata', {})

        print(f"[POLICY] Loaded policy from {policy_file}")
        return pilco, controller, metadata

    except Exception as e:
        print(f"[POLICY] ERROR loading policy: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def list_saved_policies():
    """Return all saved policy files sorted by modification time."""
    files = list(POLICY_DIR.glob("*.pkl"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def get_policy_info(policy_file=None):
    """
    Return metadata about a policy file without reconstructing the model.
    """
    if policy_file is None:
        policy_file = DEFAULT_POLICY_FILE

    if not os.path.exists(policy_file):
        return None

    try:
        with open(policy_file, 'rb') as f:
            data = pickle.load(f)

        return {
            'controller': {
                'state_dim': data['controller']['state_dim'],
                'num_basis': data['controller']['num_basis']
            },
            'pilco': {
                'horizon': data['pilco']['horizon'],
                'num_gps': len(data['pilco']['gps']),
                'training_samples': len(data['pilco']['X'])
            },
            'file_size_mb': os.path.getsize(policy_file) / (1024 * 1024),
            'experience_metadata': data.get('experience_metadata', {})
        }

    except Exception as e:
        print(f"[POLICY] Error reading policy info: {e}")
        return None
