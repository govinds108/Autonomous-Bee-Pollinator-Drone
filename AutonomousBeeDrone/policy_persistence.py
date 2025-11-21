"""
Policy Persistence System for PILCO
====================================

Saves and loads learned PILCO policies, allowing the drone to continue
training from previous sessions instead of starting from scratch.
"""

import numpy as np
import pickle
import os
from pathlib import Path


# Default policy save directory
POLICY_DIR = Path("saved_policies")
POLICY_DIR.mkdir(exist_ok=True)

DEFAULT_POLICY_FILE = POLICY_DIR / "latest_policy.pkl"


def save_policy(pilco, controller, experience_storage, policy_file=None):
    """
    Save the PILCO policy, controller, and experience storage to disk.
    
    Args:
        pilco: SimplePILCO instance
        controller: RBFController instance
        experience_storage: PILCOExperienceStorage instance
        policy_file: Path to save file (default: saved_policies/latest_policy.pkl)
    
    Returns:
        Path to saved file
    """
    if policy_file is None:
        policy_file = DEFAULT_POLICY_FILE
    
    # Extract all necessary data
    policy_data = {
        # Controller parameters
        'controller': {
            'state_dim': controller.state_dim,
            'num_basis': controller.num_basis,
            'centers': controller.centers,
            'sigma': controller.sigma,
            'weights': controller.weights
        },
        
        # PILCO GP models
        'pilco': {
            'X': pilco.X,
            'U': pilco.U,
            'Y': pilco.Y,
            'horizon': pilco.horizon,
            'lr': pilco.lr,
            'gps': []
        },
        
        # GP hyperparameters for each output dimension
        'gp_hyperparams': []
    }
    
    # Save GP models (store hyperparameters and training data)
    for i, gp in enumerate(pilco.gps):
        gp_data = {
            'X': gp.X,
            'Y': gp.Y,
            'lengthscale': gp.lengthscale,
            'variance': gp.variance,
            'noise': gp.noise,
            'K': gp.K,  # Precomputed kernel matrix
            'K_inv': gp.K_inv  # Precomputed inverse
        }
        policy_data['pilco']['gps'].append(gp_data)
        policy_data['gp_hyperparams'].append({
            'lengthscale': gp.lengthscale,
            'variance': gp.variance,
            'noise': gp.noise
        })
    
    # Save experience storage metadata (not the full buffer to avoid duplication)
    # The experience storage should be saved separately
    policy_data['experience_metadata'] = {
        'total_transitions': len(experience_storage),
        'num_flights': len(set(experience_storage.flight_ids)) if hasattr(experience_storage, 'flight_ids') else 0,
        'current_flight_id': experience_storage.current_flight_id if hasattr(experience_storage, 'current_flight_id') else 0
    }
    
    # Save to file
    with open(policy_file, 'wb') as f:
        pickle.dump(policy_data, f)
    
    print(f"[POLICY] Saved policy to {policy_file}")
    return policy_file


def load_policy(policy_file=None):
    """
    Load a previously saved PILCO policy from disk.
    
    Args:
        policy_file: Path to policy file (default: saved_policies/latest_policy.pkl)
    
    Returns:
        Tuple of (pilco, controller, experience_metadata) or (None, None, None) if file doesn't exist
    """
    if policy_file is None:
        policy_file = DEFAULT_POLICY_FILE
    
    if not os.path.exists(policy_file):
        print(f"[POLICY] No saved policy found at {policy_file}")
        return None, None, None
    
    try:
        with open(policy_file, 'rb') as f:
            policy_data = pickle.load(f)
        
        # Reconstruct controller
        from rbf_controller import RBFController
        controller = RBFController(
            state_dim=policy_data['controller']['state_dim'],
            num_basis=policy_data['controller']['num_basis']
        )
        controller.centers = policy_data['controller']['centers']
        controller.sigma = policy_data['controller']['sigma']
        controller.weights = policy_data['controller']['weights']
        
        # Reconstruct PILCO
        from simple_pilco import SimplePILCO
        pilco = SimplePILCO(
            X=policy_data['pilco']['X'],
            U=policy_data['pilco']['U'],
            Y=policy_data['pilco']['Y'],
            controller=controller,
            horizon=policy_data['pilco']['horizon'],
            lr=policy_data['pilco']['lr']
        )
        
        # Reconstruct GP models
        from simple_gp import SimpleGP
        pilco.gps = []
        for gp_data in policy_data['pilco']['gps']:
            gp = SimpleGP(
                X=gp_data['X'],
                Y=gp_data['Y'],
                lengthscale=gp_data['lengthscale'],
                variance=gp_data['variance'],
                noise=gp_data['noise']
            )
            # Restore precomputed kernel matrices if available
            if 'K' in gp_data and 'K_inv' in gp_data:
                gp.K = gp_data['K']
                gp.K_inv = gp_data['K_inv']
            else:
                # Recompute if not saved
                gp._compute_kernel()
            pilco.gps.append(gp)
        
        # Get experience metadata
        experience_metadata = policy_data.get('experience_metadata', {})
        
        print(f"[POLICY] Loaded policy from {policy_file}")
        print(f"[POLICY] Controller: {controller.state_dim}D state, {controller.num_basis} basis functions")
        print(f"[POLICY] PILCO: {len(pilco.gps)} GP models, horizon={pilco.horizon}")
        
        return pilco, controller, experience_metadata
    
    except Exception as e:
        print(f"[POLICY] ERROR loading policy: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def list_saved_policies():
    """List all saved policy files."""
    policies = list(POLICY_DIR.glob("*.pkl"))
    return sorted(policies, key=lambda p: p.stat().st_mtime, reverse=True)


def get_policy_info(policy_file=None):
    """Get information about a saved policy without fully loading it."""
    if policy_file is None:
        policy_file = DEFAULT_POLICY_FILE
    
    if not os.path.exists(policy_file):
        return None
    
    try:
        with open(policy_file, 'rb') as f:
            policy_data = pickle.load(f)
        
        info = {
            'controller': {
                'state_dim': policy_data['controller']['state_dim'],
                'num_basis': policy_data['controller']['num_basis']
            },
            'pilco': {
                'horizon': policy_data['pilco']['horizon'],
                'num_gps': len(policy_data['pilco']['gps']),
                'training_samples': len(policy_data['pilco']['X'])
            },
            'file_size_mb': os.path.getsize(policy_file) / (1024 * 1024),
            'experience_metadata': policy_data.get('experience_metadata', {})
        }
        return info
    except Exception as e:
        print(f"[POLICY] Error reading policy info: {e}")
        return None
