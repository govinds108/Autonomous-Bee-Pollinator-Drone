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

    controller = RBFController(state_dim=X.shape[1], num_basis=10)
    pilco = SimplePILCO(X, U, Y, controller, horizon=30, lr=0.01)

    start_state = X[0]
    controller = pilco.optimize(start_state, iters=20)

    return pilco, controller
