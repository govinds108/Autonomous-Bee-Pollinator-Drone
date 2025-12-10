import pickle
import numpy as np
from simple_pilco import SimplePILCO
from rbf_controller import RBFController
from pilco_experience_storage import PILCOExperienceStorage

# Load experience
experience = PILCOExperienceStorage(storage_file="saved_experience/experience.pkl")
X, U, Y = experience.get_training_arrays()

state_dim = X.shape[1]        # should be 2
action_dim = U.shape[1]       # MUST be 2 → [yaw, forward]

controller = RBFController(
    state_dim=state_dim,
    num_basis=15,
    action_dim=action_dim
)

start_state = X[0]

pilco = SimplePILCO(X, U, Y, controller, horizon=30, lr=0.01)

print("Training PILCO…")
controller = pilco.optimize(start_state, iters=40)

# Save controller + model
with open("saved_experience/pilco_model.pkl", "wb") as f:
    pickle.dump(pilco, f)

with open("saved_experience/controller.pkl", "wb") as f:
    pickle.dump(controller, f)

print("Training complete. Policy saved.")
