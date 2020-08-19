from jax.config import config
config.update("jax_enable_x64", True)

from jax import numpy as np
from ProbGeo.solvers import shooting_geodesic_solver
from jax import random


class FakeSVGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    cov_weight = 38.
    cov_weight = 1.
    cov_weight = 0.15
    cov_weight = 0.35
    # cov_weight = 0.15
    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    Y = a_mu
    cov_fn = kernel.K


class FakeGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    # cov_weight = 38.
    # cov_weight = 1.
    # cov_weight = 0.15
    cov_weight = 0.35
    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    Y = a_mu
    cov_fn = kernel.K


class FakeODE:
    from ProbGeo.metric_tensor import gp_metric_tensor
    gp = FakeGP()
    pos_init = np.array([2., -2.2])
    pos_end_targ = np.array([-1.5, 2.8])
    # pos_end_targ = np.array([-0.5, 2.8])
    # vel_init_guess = np.array([-2.5999999, 2.90100137])
    # vel_init_guess = np.array([-5.5999999, 5.90100137])
    # vel_init_guess = np.array([-15.5999999, 2.90100137])
    vel_init_guess = np.array([-5.641, 3.95])
    # vel_init_guess = np.array([-5.7, 3.])
    # vel_init_guess = np.array([-0.7, 3.])
    state_init = np.concatenate([pos_init, vel_init_guess])
    metric_fn = gp_metric_tensor
    # metric_fn_args = (gp.cov_fn, gp.X, gp.Y, gp.cov_weight)
    metric_fn_args = [gp.cov_fn, gp.X, gp.Y, gp.cov_weight]
    # metric_fn_args = [gp.cov_fn, gp.X, gp.Y]
    num_timesteps = 100
    times = np.linspace(0., 1., num_timesteps)
    t_init = 0.
    t_end = 1.
    int_method = 'Radau'
    # int_method = 'RK45'


def test_shooting_geodesic_solver():
    from ProbGeo.metric_tensor import gp_metric_tensor
    ode = FakeODE()
    metric_fn = gp_metric_tensor
    root_tol = 0.0005
    maxfev = 1000
    opt_vel_init, geodesic_traj = shooting_geodesic_solver(
        ode.pos_init, ode.pos_end_targ, ode.vel_init_guess, metric_fn,
        ode.metric_fn_args, ode.times, ode.t_init, ode.t_end, ode.int_method,
        root_tol, maxfev)
    return opt_vel_init, geodesic_traj


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict

    # Plot manifold with start and end points
    test_inputs, xx, yy = create_grid(FakeGP.X, N=961)
    mu, cov = gp_predict(test_inputs, FakeGP.X, FakeGP.Y, FakeGP.kernel)
    var = np.diag(cov).reshape(-1, 1)
    fig, axs = plot_mean_and_var(xx, yy, mu, var)
    for ax in axs:
        ax.scatter(FakeODE.pos_init[0],
                   FakeODE.pos_init[1],
                   marker='o',
                   color='r')
        ax.scatter(FakeODE.pos_end_targ[0],
                   FakeODE.pos_end_targ[1],
                   color='r',
                   marker='o')
        ax.annotate("start", (FakeODE.pos_init[0], FakeODE.pos_init[1]))
        ax.annotate("end", (FakeODE.pos_end_targ[0], FakeODE.pos_end_targ[1]))
    # plt.show()

    opt_vel_init, geodesic_traj = test_shooting_geodesic_solver()

    for ax in axs:
        ax.scatter(geodesic_traj[0, :],
                   geodesic_traj[1, :],
                   marker='x',
                   color='k')
        ax.plot(geodesic_traj[0, :],
                geodesic_traj[1, :],
                marker='x',
                color='k')
    plt.show()
