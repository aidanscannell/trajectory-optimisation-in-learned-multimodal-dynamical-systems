import pickle

import matplotlib.pyplot as plt
from gpjax.kernels import RBF
from jax import numpy as np
from jax.config import config
from ProbGeo.collocation import collocation
from ProbGeo.gp_old.kernels import DiffRBF
from ProbGeo.metric_tensor import gp_metric_tensor

config.update("jax_enable_x64", True)


def init_gating_svgp_from_saved_mogpe(
    expert_num="0",
    save_model_dir="./trained-mogpe-params/09-08-144846-param_dict.pickle",
):
    f = open(save_model_dir, "rb")
    param_dict = pickle.load(f)
    # print(param_dict.keys())
    q_mu = param_dict[".gating_network.gating_function_list[0].q_mu"].numpy()
    output_dim = q_mu.shape[1]

    q_mu = param_dict[
        ".gating_network.gating_function_list[" + expert_num + "].q_mu"
    ].numpy()
    q_sqrt = param_dict[
        ".gating_network.gating_function_list[" + expert_num + "].q_sqrt"
    ].numpy()
    Z = param_dict[
        ".gating_network.gating_function_list["
        + expert_num
        + "].inducing_variable.inducing_variable.Z"
    ].numpy()

    noise_vars = []
    k = 0
    while True:
        try:
            noise_var = param_dict[
                ".experts.experts_list[" + str(k) + "].likelihood.variance"
            ].numpy()
            # print("noise var = ", noise_var)
            noise_vars.append(noise_var)
            k += 1
        except:
            break

    kernels, mean_funcs = [], []
    for i in range(output_dim):
        variance = param_dict[
            ".gating_network.gating_function_list["
            + expert_num
            + "].kernel.kernels["
            + str(i)
            + "].variance"
        ].numpy()
        # print("var ", i)
        # print(variance)
        lengthscales = param_dict[
            ".gating_network.gating_function_list["
            + expert_num
            + "].kernel.kernels["
            + str(i)
            + "].lengthscales"
        ].numpy()  # [2]
        # mean_func = param_dict['.gating_network.gating_function_list[' +
        #                        expert_num + '].mean_function.c'].numpy()
        mean_func = 0.0

        kernel = DiffRBF(
            Z.shape[1], variance=variance, lengthscale=lengthscales, ARD=True
        )
        # kernel = RBF(variance=variance, lengthscales=lengthscales)
        mean_funcs.append(mean_func)
        kernels.append(kernel)
    return Z, q_mu, q_sqrt, kernels, mean_funcs, noise_vars


def create_save_dir():
    import pathlib
    import time

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
    # Initial guess of velocity at each collocation point
    if vel_init_guess is None:
        # TODO dynamically calculate vel_init_guess
        vel_init_guess = np.array([0.0000005, 0.0000003])
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)
    return state_guesses


# save_model_dir = "./trained-mogpe-params/09-08-144846-param_dict.pickle"
save_model_dir = "~/Developer/python-projects/mogpe/examples/logs/quadcopter/two_experts/01-19-121349"

# How may collocation points to use in solver
num_col_points = 10
input_dim = 2

# Initialise solver times
t_init = -1.0
# t_init = 0.
t_end = 1.0
times = np.linspace(t_init, t_end, num_col_points)

# Specify the desired start and end states
pos_init = np.array([0.7, -2.3])
pos_end_targ = np.array([1.0, 2.2])

# Initial guess of state vector at collocation points
# Guess a straight line trajectory
vel_init_guess = np.array([0.0000005, 0.0000003])
state_guesses = init_straight_trajectory(
    pos_init, pos_end_targ, vel_init_guess
)


# Init SVGP from saved mogpe model
# X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
(
    Z,
    q_mu,
    q_sqrt,
    kernel,
    mean_func,
    noise_vars,
) = init_gating_svgp_from_saved_mogpe(save_model_dir=save_model_dir)

metric_fn_kwargs = {
    "X": Z,
    "kernel": kernel,
    "mean_func": mean_func,
    "f": q_mu,
    "full_cov": True,
    "q_sqrt": q_sqrt,
    "cov_weight": 0.5,
    "jitter": 1e-4,
    "white": True,
}


geodesic_traj = collocation(
    state_guesses=state_guesses,
    pos_init=pos_init,
    pos_end_targ=pos_end_targ,
    metric_fn=gp_metric_tensor,
    metric_fn_kwargs=metric_fn_kwargs,
    times=times,
)


# geodesic_traj = collocation_root(solver.state_guesses, solver.pos_init,
#                                  solver.pos_end_targ, metric_fn,
#                                  metric.metric_fn_kwargs, solver.times)
# geodesic_traj = collocation_(solver.pos_init, solver.pos_end_targ,
#                              metric_fn, metric.metric_fn_kwargs,
#                              solver.times)
