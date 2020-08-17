from jax.interpreters import batching
from jax import linear_util as lu
from jax import jacfwd, jit, jvp
from jax import numpy as np
from jax import partial, tree_map, vmap
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
    grad_vec_metric_tensor_wrt_pos = grad_func(pos, metric_fn, metric_fn_args)
    grad_vec_metric_tensor_wrt_pos = grad_vec_metric_tensor_wrt_pos.reshape(
        input_dim, input_dim * input_dim)

    metric_tensor, _, _ = metric_fn(pos, *metric_fn_args)
    inv_metric_tensor = np.linalg.inv(metric_tensor)

    acc = -0.5 * inv_metric_tensor @ grad_vec_metric_tensor_wrt_pos @ kron_vel
    acc = acc.reshape(1, input_dim)

    state_prime = np.concatenate([vel, acc], -1)
    return state_prime.reshape(-1)


# class FakeGP:
#     from ProbGeo.utils.gp import load_data_and_init_kernel_fake
#     cov_weight = 1.
#     X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
#         filename='./saved_models/params_fake.npz')
#     Y = a_mu
#     cov_fn = kernel.K

# class FakeODE:
#     from ProbGeo.metric_tensor import gp_metric_tensor
#     gp = FakeGP()
#     pos_init = np.array([2., -2.2])
#     pos_end_targ = np.array([-1.5, 2.8])
#     vel_init_guess = np.array([-5.6, 3.9])
#     state_init = np.concatenate([pos_init, vel_init_guess])
#     metric_fn = gp_metric_tensor
#     # metric_fn_args = (gp.cov_fn, gp.X, gp.Y, gp.cov_weight)
#     metric_fn_args = [gp.cov_fn, gp.X, gp.Y, gp.cov_weight]
#     times = np.linspace(0, 1, 10)
#     t_init = 0.
#     t_end = 1.

# # TODO test sparse and non sparse cov_fn
# def test_geodesic_ode():
#     from ProbGeo.metric_tensor import gp_metric_tensor
#     ode = FakeODE()
#     metric_fn = gp_metric_tensor
#     state_prime = geodesic_ode(ode.t_init, ode.state_init, metric_fn,
#                                ode.metric_fn_args)

if __name__ == "__main__":

    # test_geodesic_ode()
    test_shooting_geodesic_solver()
