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
print("optimised_trajectory")
print(optimised_trajectory)
print(optimised_trajectory.shape)

# Configure environment from toml config file and set delta_time from solver
toml_env_config_filename = "../../../../sim-envs/scenario-1/configs/env_config.toml"  # environment config to use
gating_bitmap_filename = "../../../../sim-envs/scenario-1/bitmaps/gating_mask.bmp"
env = parse_toml_config_to_VelocityControlledQuadcopter2DEnv(
    toml_env_config_filename, gating_bitmap_filename=gating_bitmap_filename
)
env.delta_time = delta_time


# Simulate the optimised controls at the optimised state locations
num_traj_steps = optimised_trajectory.shape[0]
delta_state_outputs = []
for step in range(num_traj_steps):
    delta_state = transition_dynamics(optimised_trajectory[step, :], env)
    delta_state_outputs.append(delta_state)
delta_state_outputs = np.stack(delta_state_outputs)
# state_outputs = optimised_trajectory[:-1, :2] + delta_state_outputs[1:, :]


state_init = (optimised_trajectory[0:1, 0:2]).copy()
print("state init")
print(state_init)
env.reset(state_init)
env.previous_velocity = optimised_trajectory[0:1, 2:4]
simulated_trajectory = [collocation_solver.optimised_trajectory[0, :2]]
state = state_init
for step in range(num_traj_steps - 1):
    action = collocation_solver.optimised_trajectory[step + 1, 2:4].reshape(1, -1)
    state_action_input = np.concatenate([state, action])
    print("state_action")
    print(state_action_input)
    delta_state = transition_dynamics(state_action_input, env)
    state = state + delta_state
    simulated_trajectory.append(state.reshape(-1))
# print(optimised_trajectory)
# print(collocation_solver.optimised_trajectory)
simulated_trajectory = np.stack(simulated_trajectory)
print("simulated trajectory")
print(simulated_trajectory)

# Simulate the optimised controls in an open loop trajectory
# state_init = (optimised_trajectory[0:1, 0:2]).copy()
# state_init = (optimised_trajectory[1:2, 0:2]).copy()
# print("state init")
# print(state_init)
# env.reset(state_init)
# # env.reset(optimised_trajectory[0:1, 0:2])
# env.previous_velocity = optimised_trajectory[0:1, 2:4]
# simulated_trajectory = [collocation_solver.optimised_trajectory[0, :2]]
# simulated_trajectory = [collocation_solver.optimised_trajectory[1, :2]]
# for step in range(num_traj_steps - 1):
#     action = collocation_solver.optimised_trajectory[step + 1, 2:4].reshape(1, -1)
#     print("action")
#     print(action)
#     next_state = env.step(action)
#     print("next_state")
#     print(next_state.observation)
#     simulated_trajectory.append(next_state.observation.reshape(-1))
# print(optimised_trajectory)
# print(collocation_solver.optimised_trajectory)
# simulated_trajectory = np.stack(simulated_trajectory)
# print("simulated trajectory")
# print(simulated_trajectory)
# print(simulated_trajectory.shape)
simulated_delta_state_outputs = (
    simulated_trajectory[1:, :2] - simulated_trajectory[:-1, :2]
)

fig, axs = plot_svgp_mean_and_var(svgp=collocation_solver.ode.metric_tensor.gp)
for ax in axs:
    ax.quiver(
        simulated_trajectory[:-1, 0],
        simulated_trajectory[:-1, 1],
        simulated_delta_state_outputs[:, 0],
        simulated_delta_state_outputs[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        # alpha=0.4
        # label="Simulated delta state $\Delta\mathbf{x}$",
    )

    # ax.scatter(
    #     collocation_solver.optimised_trajectory[0, 0],
    #     collocation_solver.optimised_trajectory[0, 1],
    #     marker="x",
    # )
    print("start")
    print(optimised_trajectory[0, :2])
    print(optimised_trajectory[-1, :2])
    # ax.annotate("Start", collocation_solver.optimised_trajectory[0, :2])
    # ax.annotate("End", collocation_solver.optimised_trajectory[-1, :2])
    ax.annotate("Start", optimised_trajectory[0, :2])
    # ax.annotate("End", optimised_trajectory[-1, :2])
    ax.scatter(
        collocation_solver.optimised_trajectory[:, 0],
        collocation_solver.optimised_trajectory[:, 1],
        marker="+",
        # alpha=0.4,
        linewidths=0.6,
        label="Optimised state trajectory $\mathbf{x}^{*}_{1:T}$",
    )
    # ax.plot(state_outputs[:, 0], state_outputs[:, 1], marker="o", label="")
    ax.scatter(
        # simulated_trajectory[:-1, 0],
        # simulated_trajectory[:-1, 1],
        simulated_trajectory[:, 0],
        simulated_trajectory[:, 1],
        marker="x",
        # alpha=0.4,
        linewidths=0.6,
        label="Simulated state trajectory $\{ \mathbf{x}_{t} =  f_{sim}(\mathbf{x}_{t-1}, \mathbf{u}^*_{t-1})) \mid \mathbf{u}^*_t \in \mathbf{u}^*_{1:T} \}$",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.quiver(
        collocation_solver.optimised_trajectory[:-1, 0],
        collocation_solver.optimised_trajectory[:-1, 1],
        delta_state_outputs[1:, 0],
        delta_state_outputs[1:, 1],
        angles="xy",
        scale_units="xy",
        width=0.004,
        scale=1.0,
        label="Simulated delta state $\Delta\mathbf{x}$",
    )
axs[0].legend(loc=3)
save_filename = "./images/simulated-trajectory-over-svgp.pdf"
plt.savefig(save_filename, transparent=True, bbox_inches="tight")
# plt.show()


# def length(traj, metric_tensor):
#     def segment_length(pos, vel):
#         metric = metric_tensor(pos)
#         inner_prod = vel.T @ metric @ vel
#         print("inner prod")
#         print(inner_prod)
#         return inner_prod
#         # sqrt_inner_prod = jnp.sqrt(inner_prod)
#         # print(sqrt_inner_prod)
#         # return sqrt_inner_prod

#     posses = traj[:, 0:2]
#     vels = traj[:, 2:4]
#     length_segment = jax.vmap(segment_length)(posses, vels)
#     length_total = jnp.sum(length_segment)
#     return length_total
# traj_length = length(traj, metric_tensor=metric_tensor)
# print("traj length")
# print(traj_length)
