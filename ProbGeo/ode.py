from jax import jacfwd, jit, jvp, partial, tree_map, vmap
from jax.interpreters import batching
from jax import linear_util as lu
from jax import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from ProbGeo.metric_tensor import calc_vec_metric_tensor


@partial(jit, static_argnums=(2, 3))
def geodesic_ode(t, state, metric_fn, metric_fn_args):
    input_dim = int(state.shape[0] / 2)
    pos = state[:input_dim].reshape(1, -1)
    vel = state[input_dim:].reshape(1, -1)
    kron_vel = np.kron(vel, vel).T

    # TODO implement value and jac
    grad_func = jacfwd(calc_vec_metric_tensor, 0)
    grad_vec_metric_tensor_wrt_pos = grad_func(pos.reshape(-1), metric_fn,
                                               metric_fn_args)
    # grad_vec_metric_tensor_wrt_pos = grad_vec_metric_tensor_wrt_pos.reshape(
    #     input_dim, input_dim * input_dim)
    print('grad vec')
    print(grad_vec_metric_tensor_wrt_pos.shape)
    grad_vec_metric_tensor_wrt_posT = grad_vec_metric_tensor_wrt_pos.T

    metric_tensor, _, _ = metric_fn(pos, *metric_fn_args)
    # TODO implement cholesky if metric_tensor is PSD
    inv_metric_tensor = np.linalg.inv(metric_tensor)

    acc = -0.5 * inv_metric_tensor @ grad_vec_metric_tensor_wrt_posT @ kron_vel
    acc = acc.reshape(1, input_dim)

    state_prime = np.concatenate([vel, acc], -1)
    return state_prime.reshape(-1)


@partial(jit, static_argnums=(2, 3))
def geodesic_ode_bvp(t, state, metric_fn, metric_fn_args):
    state_prime = vmap(geodesic_ode,
                       in_axes=(None, 1, None, None))(t, state, metric_fn,
                                                      metric_fn_args)
    state_primeT = state_prime.T
    return state_primeT
