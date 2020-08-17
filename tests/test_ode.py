from jax import numpy as np
from ProbGeo.ode import geodesic_ode


class FakeGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    cov_weight = 1.
    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    Y = a_mu
    cov_fn = kernel.K


class FakeODE:
    from ProbGeo.metric_tensor import gp_metric_tensor
    gp = FakeGP()
    pos_init = np.array([2., -2.2])
    pos_end_targ = np.array([-1.5, 2.8])
    vel_init_guess = np.array([-5.6, 3.9])
    state_init = np.concatenate([pos_init, vel_init_guess])
    metric_fn = gp_metric_tensor
    metric_fn_args = [gp.cov_fn, gp.X, gp.Y, gp.cov_weight]
    times = np.linspace(0, 1, 10)
    t_init = 0.
    t_end = 1.


# TODO test sparse and non sparse cov_fn
def test_geodesic_ode():
    from ProbGeo.metric_tensor import gp_metric_tensor
    ode = FakeODE()
    metric_fn = gp_metric_tensor
    state_prime = geodesic_ode(ode.t_init, ode.state_init, metric_fn,
                               ode.metric_fn_args)


if __name__ == "__main__":

    test_geodesic_ode()
