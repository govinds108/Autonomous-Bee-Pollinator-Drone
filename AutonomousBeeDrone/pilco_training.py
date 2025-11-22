import numpy as np
from rbf_controller import RBFController
from simple_pilco import SimplePILCO

def prepare_pilco_data(S, A, S2):
    X = S[:-1]
    U = A[:-1]
    Y = (S2 - S)[:-1]
    return X, U, Y

def train_pilco(S, A, S2):
    X, U, Y = prepare_pilco_data(S, A, S2)

    # Subsample data if dataset is large to keep GP training tractable.
    # Large GP kernel inversions scale cubically and can exhaust memory/CPU.
    max_samples = 200
    if X.shape[0] > max_samples:
        # even spacing preserves temporal coverage
        idx = np.linspace(0, X.shape[0]-1, max_samples).astype(int)
        X = X[idx]
        U = U[idx]
        Y = Y[idx]

    controller = RBFController(state_dim=X.shape[1], num_basis=10)
    pilco = SimplePILCO(X, U, Y, controller, horizon=30, lr=0.01)

    start_state = X[0]
    controller = pilco.optimize(start_state, iters=20)

    return pilco, controller
