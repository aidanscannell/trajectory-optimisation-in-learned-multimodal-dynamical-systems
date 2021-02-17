import pathlib
import time

import gpjax
import jax.numpy as jnp
import numpy as np
from gpjax.kernels import RBF
from gpjax.models import SVGP

from mogpe.training.utils import load_model_from_config_and_checkpoint


def init_svgp_gpjax_from_mogpe_ckpt(
    config_file, ckpt_dir, data_file, expert_num=1
) -> SVGP:
    """Initialise a gpjax.models.SVGP from an mogpe.mixture_models.MixtureOfSVGPExperts ckpt

    Initialise a gpjax.models.SVGP with parameters from an SVGP gating function from a
    mogpe.mixture_models.MixtureOfSVGPExperts training checkpoint.

    :param config_file: TOML config file for MixtureOfSVGPExperts model
    :param ckpt_dir: path to saved checkpoints for MixtureOfSVGPExperts model
    :param data_file: path to the training data set as a numpy file (.npz)
    :param expert_num: expert indicator variable (k = {1,...,K})
    :returns: Instance of gpjax.models.SVGP
    """
    likelihood = None
    num_latent_gps = 1
    # load data set
    data = jnp.load(data_file)
    X = data["x"]

    # configure mogpe model from checkpoint
    model = load_model_from_config_and_checkpoint(config_file, ckpt_dir, X=X)

    # select the gating function to use
    gating_func = model.gating_network.gating_function_list[expert_num - 1]
    mean_function = (
        gpjax.mean_functions.Zero()
    )  # mogpe gating functions have zero mean function
    # mean_function = 0.0  # mogpe gating functions have zero mean function
    whiten = gating_func.whiten

    # sparse GP parameters
    q_diag = gating_func.q_diag
    q_mu = gating_func.q_mu.numpy()
    q_sqrt = gating_func.q_sqrt.numpy()
    inducing_variable = gating_func.inducing_variable.inducing_variable.Z.numpy()

    # kerenl parameters
    variance = gating_func.kernel.kernels[0].variance.numpy()
    lengthscales = gating_func.kernel.kernels[0].lengthscales.numpy()
    kernel = RBF(variance=variance, lengthscales=lengthscales)

    svgp = SVGP(
        kernel,
        likelihood,
        inducing_variable,
        mean_function,
        num_latent_gps,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    return svgp


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


def init_straight_trajectory(
    pos_init,
    pos_end,
    vel_init_guess=None,
    num_col_points=10,
    endpoint=True,
):
    pos_dim = pos_init.shape[0]
    pos1_guesses = jnp.linspace(
        pos_init[0], pos_end[0], num_col_points, endpoint=endpoint
    )
    pos2_guesses = jnp.linspace(
        pos_init[1], pos_end[1], num_col_points, endpoint=endpoint
    )
    pos_guesses = jnp.stack([pos1_guesses, pos2_guesses], -1)

    # pos_guesses = np.random.uniform(
    #     pos_init,
    #     pos_end_targ,
    #     (num_col_points, pos_dim),
    # )
    # Initial guess of velocity at each collocation point
    if vel_init_guess is None:
        # TODO dynamically calculate vel_init_guess
        vel_init_guess = jnp.array([0.0000005, 0.0000003])
    vel_guesses = jnp.broadcast_to(vel_init_guess, (num_col_points, pos_dim))
    state_guesses = jnp.concatenate([pos_guesses, vel_guesses], -1)
    return state_guesses


def init_two_segment_straight_trajectory(
    pos_init, pos_mid, pos_end, vel_init_guess=None, num_col_points=10
):
    num_col_points = int(num_col_points / 2)
    state_guesses_1 = init_straight_trajectory(
        pos_init,
        pos_mid,
        vel_init_guess=vel_init_guess,
        num_col_points=num_col_points,
        endpoint=False,
    )

    state_guesses_2 = init_straight_trajectory(
        pos_mid,
        pos_end,
        vel_init_guess=vel_init_guess,
        num_col_points=num_col_points,
    )
    state_guesses = jnp.concatenate([state_guesses_1, state_guesses_2], 0)
    return state_guesses


def init_start_end_pos_scipy_bounds(
    opt_vars, pos_init, pos_end, pos_init_idx=0, pos_end_idx=-1, tol=0.02
):
    # disable bounds on all variables
    lb = -jnp.ones([*opt_vars.shape]) * jnp.inf
    ub = jnp.ones([*opt_vars.shape]) * jnp.inf

    def update_bound_at_idx(lb, ub, pos_at_idx, idx, tol=0.02):
        l_tol = 1 - tol
        u_tol = 1 + tol
        for i, pos in enumerate(pos_at_idx):
            if pos < 0:
                lb = jax.ops.index_update(lb, jax.ops.index[idx, i], pos * u_tol)
                ub = jax.ops.index_update(ub, jax.ops.index[idx, i], pos * l_tol)
            else:
                lb = jax.ops.index_update(lb, jax.ops.index[idx, i], pos * l_tol)
                ub = jax.ops.index_update(ub, jax.ops.index[idx, i], pos * u_tol)
        return lb, ub

    # add bounds for start and end positions
    lb, ub = update_bound_at_idx(lb, ub, pos_init, idx=pos_init_idx, tol=tol)
    lb, ub = update_bound_at_idx(lb, ub, pos_end, idx=pos_end_idx, tol=tol)
    bounds = Bounds(lb=lb.flatten(), ub=ub.flatten())
    return bounds
