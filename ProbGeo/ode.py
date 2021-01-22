from jax import jacfwd, jit
from jax import numpy as np
from jax import partial, vmap

from ProbGeo.metric_tensor import calc_vec_metric_tensor


# @partial(jit, static_argnums=(2))
# @partial(jit, static_argnums=(2, 3))
@partial(jit, static_argnums=(2, 3))
def geodesic_ode(t, state, metric_fn, metric_fn_kwargs):
    print("inside geodesic_ode")
    print(state.shape)
    input_dim = int(state.shape[0] / 2)
    pos = state[:input_dim].reshape(1, -1)
    vel = state[input_dim:].reshape(1, -1)
    kron_vel = np.kron(vel, vel).T

    # TODO implement value and jac
    grad_func = jacfwd(calc_vec_metric_tensor, 0)
    grad_vec_metric_tensor_wrt_pos = grad_func(
        pos.reshape(-1), metric_fn, metric_fn_kwargs
    )
    # grad_vec_metric_tensor_wrt_pos = grad_vec_metric_tensor_wrt_pos.reshape(
    #     input_dim, input_dim * input_dim)
    print("grad vec")
    print(grad_vec_metric_tensor_wrt_pos.shape)
    grad_vec_metric_tensor_wrt_posT = grad_vec_metric_tensor_wrt_pos.T

    try:
        metric_tensor, _ = metric_fn(pos, **metric_fn_kwargs)
    except:
        metric_tensor, _, _ = metric_fn(pos, **metric_fn_kwargs)
    # TODO implement cholesky if metric_tensor is PSD
    jitter = 1e-6
    metric_tensor += np.eye(input_dim) * jitter
    inv_metric_tensor = np.linalg.inv(metric_tensor)
    print("inv")
    print(inv_metric_tensor)
    # assert np.isnan(inv_metric_tensor).any()

    acc = -0.5 * inv_metric_tensor @ grad_vec_metric_tensor_wrt_posT @ kron_vel
    acc = acc.reshape(1, input_dim)

    state_prime = np.concatenate([vel, acc], -1)
    print("state_prime")
    print(state_prime)
    # return state_prime
    return state_prime.reshape([-1])
    # return state_prime.reshape(-1)


@partial(jit, static_argnums=(2, 3))
def geodesic_ode_bvp(t, state, metric_fn, metric_fn_kwargs):
    state_prime = vmap(geodesic_ode, in_axes=(None, 1, None, None))(
        t, state, metric_fn, metric_fn_kwargs
    )
    state_primeT = state_prime.T
    return state_primeT
