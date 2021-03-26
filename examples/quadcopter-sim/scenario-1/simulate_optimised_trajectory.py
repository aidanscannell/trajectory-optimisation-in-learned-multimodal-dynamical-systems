from tromp.plotting.gp import plot_svgp_mean_and_var
import pickle
import matplotlib.pyplot as plt
import numpy as np
from simenvs.generate_dataset_from_env import transition_dynamics
from simenvs.parse_env_toml_config import (
    parse_toml_config_to_VelocityControlledQuadcopter2DEnv,
)

params = {
    "text.usetex": True,
    "text.latex.preamble": [
        "\\usepackage{amssymb}",
        "\\usepackage{amsmath}",
    ],
}
plt.rcParams.update(params)

# Load saved solver
solver_path = "./saved-trajectories/GeodesicSolver.pickle"
collocation_solver = pickle.load(open(solver_path, "rb", -1))
optimised_trajectory = collocation_solver.optimised_trajectory.copy()
delta_time = collocation_solver.times[1] - collocation_solver.times[0]
print("Optimised Trajectory")
print(optimised_trajectory)

# Configure environment from toml config file and set delta_time from solver
toml_env_config_filename = "../../../../sim-envs/scenario-1/configs/env_config.toml"  # environment config to use
gating_bitmap_filename = "../../../../sim-envs/scenario-1/bitmaps/gating_mask.bmp"
env = parse_toml_config_to_VelocityControlledQuadcopter2DEnv(
    toml_env_config_filename, gating_bitmap_filename=gating_bitmap_filename
)
env.delta_time = delta_time

# Simulate the optimised controls in an open loop
state = (optimised_trajectory[0:1, 0:2]).copy()
env.reset(state)
env.previous_velocity = optimised_trajectory[0:1, 2:4]
simulated_trajectory = [collocation_solver.optimised_trajectory[0, :2]]
num_traj_steps = optimised_trajectory.shape[0]
for step in range(num_traj_steps - 1):
    action = collocation_solver.optimised_trajectory[step + 1, 2:4].reshape(1, -1)
    state_action_input = np.concatenate([state, action])
    delta_state = transition_dynamics(state_action_input, env)
    state = state + delta_state
    simulated_trajectory.append(state.reshape(-1))
simulated_trajectory = np.stack(simulated_trajectory)
print("Simulated Trajectory")
print(simulated_trajectory)
# simulated_delta_state_outputs = simulated_trajectory[1:, :2] - simulated_trajectory[:-1, :2])

fig, axs = plot_svgp_mean_and_var(svgp=collocation_solver.ode.metric_tensor.gp)
for ax in axs:
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.annotate("Start", optimised_trajectory[0, :2])
    # ax.annotate("End", optimised_trajectory[-1, :2])

    # ax.quiver(
    #     simulated_trajectory[:-1, 0],
    #     simulated_trajectory[:-1, 1],
    #     simulated_delta_state_outputs[:, 0],
    #     simulated_delta_state_outputs[:, 1],
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1.0,
    #     width=0.004,
    # )

    ax.scatter(
        collocation_solver.optimised_trajectory[:, 0],
        collocation_solver.optimised_trajectory[:, 1],
        marker="+",
        color="k",
        # alpha=0.4,
        linewidths=0.6,
        label="Optimised state trajectory $\mathbf{x}^{*}_{1:T}$",
    )
    ax.plot(
        simulated_trajectory[:, 0],
        simulated_trajectory[:, 1],
        marker="x",
        color="darkorange",
        linewidth=0.6,
        markersize=3.0,
        # linewidths=0.6,
        label="Simulated state trajectory $\{ \mathbf{x}_{t} \mid \Delta \mathbf{x}_{t} =  f_{sim}(\mathbf{x}_{t-1}, \mathbf{u}^*_{t-1})),  \mathbf{u}^*_t \in \mathbf{u}^*_{1:T} \}$",
    )

axs[0].legend(loc=3)
save_filename = "./images/simulated-trajectory-over-svgp.pdf"
plt.savefig(save_filename, transparent=True, bbox_inches="tight")
# plt.show()
