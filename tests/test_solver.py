from jax.config import config
config.update("jax_enable_x64", True)

from jax import numpy as np
from ProbGeo.solvers import shooting_geodesic_solver


class FakeGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    X, Y, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    mean_func = 0.


class FakeSVGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse
    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../models/saved_models/params_fake_sparse_20-08_2.npz')


class FakeGPMetric:
    gp = FakeGP()
    # cov_weight = 38.
    # cov_weight = 1.
    cov_weight = 0.35
    full_cov = True
    q_sqrt = None
    jitter = 1e-4
    white = True
    metric_fn_kwargs = {
        "X": gp.X,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.Y,
        "full_cov": full_cov,
        "q_sqrt": q_sqrt,
        "cov_weight": cov_weight,
        "jitter": jitter,
        "white": white
    }


class FakeSVGPMetric:
    gp = FakeSVGP()
    # cov_weight = 38.
    cov_weight = 50.
    cov_weight = 500.
    full_cov = True
    jitter = 1e-4
    white = True
    metric_fn_kwargs = {
        "X": gp.Z,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.q_mu,
        "full_cov": full_cov,
        "q_sqrt": gp.q_sqrt,
        "cov_weight": cov_weight,
        "jitter": jitter,
        "white": white
    }


class FakeODESolver:
    num_timesteps = 100
    t_init = 0.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_timesteps)
    int_method = 'Radau'
    # int_method = 'RK45'
    root_tol = 0.0005
    maxfev = 1000
    # pos_init = np.array([2., -2.2])
    # pos_init = np.array([2.2, -2.8])
    pos_init = np.array([2.8, -2.5])
    pos_end_targ = np.array([-1.5, 2.8])
    # pos_end_targ = np.array([-0.5, 2.8])
    # vel_init_guess = np.array([-2.5999999, 2.90100137])
    # vel_init_guess = np.array([-5.5999999, 5.90100137])
    # vel_init_guess = np.array([-15.5999999, 2.90100137])
    vel_init_guess = np.array([-5.641, 3.95])
    # vel_init_guess = np.array([-5.7, 3.])
    # vel_init_guess = np.array([-0.7, 3.])
    state_init = np.concatenate([pos_init, vel_init_guess])


class FakeODESolverSVGP:
    num_timesteps = 1000
    t_init = 0.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_timesteps)
    int_method = 'Radau'
    # int_method = 'RK45'
    root_tol = 0.0005
    maxfev = 1000
    pos_init = np.array([-1.5, -1.])
    pos_init = np.array([-2., -0.25])
    pos_end_targ = np.array([1.8, 1.8])
    vel_init_guess = np.array([-5, -3])
    state_init = np.concatenate([pos_init, vel_init_guess])


def test_shooting_geodesic_solver():
    from ProbGeo.metric_tensor import gp_metric_tensor
    solver = FakeODESolver()
    metric = FakeGPMetric()
    metric_fn = gp_metric_tensor
    opt_vel_init, geodesic_traj = shooting_geodesic_solver(
        solver.pos_init, solver.pos_end_targ, solver.vel_init_guess, metric_fn,
        metric.metric_fn_kwargs, solver.times, solver.t_init, solver.t_end,
        solver.int_method, solver.root_tol, solver.maxfev)
    return opt_vel_init, geodesic_traj


def test_shooting_geodesic_solver_with_svgp():
    from ProbGeo.metric_tensor import gp_metric_tensor
    solver = FakeODESolverSVGP()
    metric = FakeSVGPMetric()
    metric_fn = gp_metric_tensor
    opt_vel_init, geodesic_traj = shooting_geodesic_solver(
        solver.pos_init, solver.pos_end_targ, solver.vel_init_guess, metric_fn,
        metric.metric_fn_kwargs, solver.times, solver.t_init, solver.t_end,
        solver.int_method, solver.root_tol, solver.maxfev)
    return opt_vel_init, geodesic_traj


def test_and_plot_shooting_geodesic_solver():
    import matplotlib.pyplot as plt
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict

    # Plot manifold with start and end points
    gp = FakeGP()
    Xnew, xx, yy = create_grid(gp.X, N=961)
    # mu, var = gp_predict(Xnew, FakeGP.X, FakeGP.Y, FakeGP.kernel)
    mu, var = gp_predict(Xnew,
                         gp.X,
                         kernel=gp.kernel,
                         mean_func=gp.mean_func,
                         f=gp.Y,
                         full_cov=False)
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    for ax in axs:
        ax.scatter(FakeODESolver.pos_init[0],
                   FakeODESolver.pos_init[1],
                   marker='o',
                   color='r')
        ax.scatter(FakeODESolver.pos_end_targ[0],
                   FakeODESolver.pos_end_targ[1],
                   color='r',
                   marker='o')
        ax.annotate("start",
                    (FakeODESolver.pos_init[0], FakeODESolver.pos_init[1]))
        ax.annotate(
            "end",
            (FakeODESolver.pos_end_targ[0], FakeODESolver.pos_end_targ[1]))
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


def test_and_plot_shooting_geodesic_solver_with_svgp():
    import matplotlib.pyplot as plt
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict

    # Plot manifold with start and end points
    gp = FakeSVGP()
    solver = FakeODESolverSVGP()
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(
        Xnew,
        gp.Z,
        kernel=gp.kernel,
        mean_func=gp.mean_func,
        f=gp.q_mu,
        full_cov=False,
        # full_cov=True,
        q_sqrt=gp.q_sqrt)
    # var = np.diag(var)
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    for ax in axs:
        ax.scatter(solver.pos_init[0],
                   solver.pos_init[1],
                   marker='o',
                   color='r')
        ax.scatter(solver.pos_end_targ[0],
                   solver.pos_end_targ[1],
                   color='r',
                   marker='o')
        ax.annotate("start", (solver.pos_init[0], solver.pos_init[1]))
        ax.annotate("end", (solver.pos_end_targ[0], solver.pos_end_targ[1]))
    # plt.show()

    opt_vel_init, geodesic_traj = test_shooting_geodesic_solver_with_svgp()

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


if __name__ == "__main__":

    # test_and_plot_shooting_geodesic_solver()
    test_and_plot_shooting_geodesic_solver_with_svgp()
