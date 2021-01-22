import pathlib
import time

import matplotlib.pyplot as plt
from gpjax.kernels import RBF
from jax import numpy as np
from jax.config import config
from mogpe.training.utils import load_model_from_config_and_checkpoint
from tromp.collocation import collocation
from tromp.metric_tensor import gp_metric_tensor
from tromp.visualisation.plot_trajectories import plot_svgp_and_start_end

config.update("jax_enable_x64", True)


def mogpe_checkpoint_to_numpy(config_file, ckpt_dir, data_file, expert_num=0):
    # load data set
    data = np.load(data_file)
    X = data["x"]

    # configure mogpe model from checkpoint
    model = load_model_from_config_and_checkpoint(config_file, ckpt_dir, X=X)

    # select the gating function to use
    gating_func = model.gating_network.gating_function_list[expert_num]
    mean_function = 0.0  # mogpe gating functions have zero mean function
    whiten = gating_func.whiten

    # sparse GP parameters
    q_mu = gating_func.q_mu.numpy()
    q_sqrt = gating_func.q_sqrt.numpy()
    inducing_variable = (
        gating_func.inducing_variable.inducing_variable.Z.numpy()
    )

    # kerenl parameters
    variance = gating_func.kernel.kernels[0].variance.numpy()
    lengthscales = gating_func.kernel.kernels[0].lengthscales.numpy()
    kernel = RBF(variance=variance, lengthscales=lengthscales)
    return kernel, inducing_variable, mean_function, q_mu, q_sqrt, whiten


def create_save_dir():

    dir_name = (
        "../reports/figures/"
        + time.strftime("%d-%m-%Y")
        + "/"
        + time.strftime("%H%M%S")
        + "/"
    )
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    return dir_name


def init_straight_trajectory(pos_init, pos_end_targ, vel_init_guess=None):
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    input_dim = pos_init.shape[0]
    # Initial guess of velocity at each collocation point
    if vel_init_guess is None:
        # TODO dynamically calculate vel_init_guess
        vel_init_guess = np.array([0.0000005, 0.0000003])
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)
    return state_guesses


mogpe_dir = "../../../../mogpe/"
ckpt_dir = (
    mogpe_dir
    + "examples/logs/quadcopter-sim/scenario-1/two_experts/01-19-121349"
)
config_file = (
    mogpe_dir
    + "examples/quadcopter-sim/scenario-1/configs/config_2_experts.toml"
)
data_file = (
    mogpe_dir
    + "examples/quadcopter-sim/scenario-1/data/quad_sim_const_action_scenario_1.npz"
)


class SVGP:
    (
        kernel,
        inducing_variable,
        mean_function,
        q_mu,
        q_sqrt,
        whiten,
    ) = mogpe_checkpoint_to_numpy(config_file, ckpt_dir, data_file)
    Z = inducing_variable
    mean_func = mean_function


# How may collocation points to use in solver
num_col_points = 10
# input_dim = 2

# Initialise solver times
t_init = -1.0
# t_init = 0.
t_end = 1.0
times = np.linspace(t_init, t_end, num_col_points)

# Specify the desired start and end states
pos_init = np.array([2.7, 1.2])
pos_end_targ = np.array([-2.7, -0.5])
pos_init = np.array([2.7, 2.0])
pos_end_targ = np.array([-2.6, -1.5])

# Initial guess of state vector at collocation points
# Guess a straight line trajectory
vel_init_guess = np.array([0.0000005, 0.0000003])
state_guesses = init_straight_trajectory(
    pos_init, pos_end_targ, vel_init_guess
)


class CollocationSolverParams:
    times = times
    pos_init = pos_init
    pos_end_targ = pos_end_targ
    state_guesses = state_guesses


class SVGPMetric:
    gp = SVGP()
    # cov_weight = 0.5
    cov_weight = 10.0
    full_cov = True
    jitter = 1e-4
    metric_fn_kwargs = {
        "X": gp.inducing_variable,
        "kernel": gp.kernel,
        "mean_func": gp.mean_function,
        "f": gp.q_mu,
        "full_cov": full_cov,
        "q_sqrt": gp.q_sqrt,
        "cov_weight": cov_weight,
        "jitter": jitter,
        "white": gp.whiten,
    }
    # metric_fn_kwargs = {
    #     "X": gp.Z,
    #     "kernel": gp.kernel,
    #     "mean_func": gp.mean_func,
    #     "f": gp.q_mu,
    #     "full_cov": full_cov,
    #     "q_sqrt": gp.q_sqrt,
    #     "cov_weight": cov_weight,
    #     "jitter": jitter,
    #     "white": white,
    # }


gp = SVGP()
metric = SVGPMetric()
solver = CollocationSolverParams()
plot_svgp_and_start_end(gp, solver, traj_opts=None, labels=None)

plt.show()

geodesic_traj = collocation(
    state_guesses=state_guesses,
    pos_init=pos_init,
    pos_end_targ=pos_end_targ,
    metric_fn=gp_metric_tensor,
    metric_fn_kwargs=metric.metric_fn_kwargs,
    times=times,
)

plot_svgp_and_start_end(gp, solver, traj_opts=geodesic_traj)
plt.show()

# # geodesic_traj = collocation_root(solver.state_guesses, solver.pos_init,
# #                                  solver.pos_end_targ, metric_fn,
# #                                  metric.metric_fn_kwargs, solver.times)
# # geodesic_traj = collocation_(solver.pos_init, solver.pos_end_targ,
# #                              metric_fn, metric.metric_fn_kwargs,
# #                              solver.times)
