import jax
import matplotlib.pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np
from ProbGeo.solvers import shooting_geodesic_solver


def plot_gp_and_start_end(gp, solver):
    from ProbGeo.gp import gp_predict
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid

    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(
        Xnew,
        gp.X,
        kernel=gp.kernel,
        mean_func=gp.mean_func,
        f=gp.Y,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    for ax in axs:
        fig, ax = plot_start_and_end_pos(fig, ax, solver)
    return fig, axs


def plot_start_and_end_pos(fig, ax, solver):
    ax.scatter(solver.pos_init[0], solver.pos_init[1], marker="o", color="r")
    ax.scatter(
        solver.pos_end_targ[0], solver.pos_end_targ[1], color="r", marker="o"
    )
    ax.annotate("start", (solver.pos_init[0], solver.pos_init[1]))
    ax.annotate("end", (solver.pos_end_targ[0], solver.pos_end_targ[1]))
    return fig, ax


def plot_svgp_and_start_end(gp, solver):
    from ProbGeo.gp import gp_predict
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid

    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(
        Xnew,
        gp.Z,
        kernel=gp.kernel,
        mean_func=gp.mean_func,
        f=gp.q_mu,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    for ax in axs:
        fig, ax = plot_start_and_end_pos(fig, ax, solver)
    return fig, axs


def plot_traj(fig, axs, traj):
    try:
        for ax in axs:
            ax.scatter(traj[:, 0], traj[:, 1], marker="x", color="k")
            ax.plot(traj[:, 0], traj[:, 1], marker="x", color="k")
    except:
        axs.scatter(traj[:, 0], traj[:, 1], marker="x", color="k")
        axs.plot(traj[:, 0], traj[:, 1], marker="x", color="k")
    return fig, axs


def plot_mixing_prob_and_start_end(gp, solver):
    from ProbGeo.mogpe import mogpe_mixing_probability
    from ProbGeo.visualisation.gp import plot_contourf
    from ProbGeo.visualisation.utils import create_grid

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mixing_probs = mogpe_mixing_probability(
        Xnew,
        gp.X,
        gp.kernel,
        mean_func=gp.mean_func,
        f=gp.Y,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    fig, ax = plt.subplots(1, 1)
    plot_contourf(fig, ax, xx, yy, mixing_probs[:, 0:1])

    ax.scatter(solver.pos_init[0], solver.pos_init[1], marker="o", color="r")
    ax.scatter(
        solver.pos_end_targ[0], solver.pos_end_targ[1], color="r", marker="o"
    )
    ax.annotate("start", (solver.pos_init[0], solver.pos_init[1]))
    ax.annotate("end", (solver.pos_end_targ[0], solver.pos_end_targ[1]))
    return fig, ax


class FakeGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake

    X, Y, kernel = load_data_and_init_kernel_fake(
        filename="../models/saved_models/params_fake.npz"
    )
    mean_func = 0.0
    q_sqrt = None


class FakeSVGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse

    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        # filename='../models/saved_models/params_fake_sparse_26-08.npz')
        filename="../models/saved_models/params_from_model.npz"
    )
    # filename='../models/saved_models/params_fake_sparse_20-08.npz')
    # filename='../models/saved_models/params_fake_sparse_20-08_2.npz')


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
        "white": white,
    }


class FakeSVGPMetric:
    gp = FakeSVGP()
    # cov_weight = 38.
    cov_weight = 50.0
    # cov_weight = 500.
    # cov_weight = 0.
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
        "white": white,
    }


class FakeGPMetricProb:
    from ProbGeo.mogpe import single_mogpe_mixing_probability

    gp = FakeGP()
    # cov_weight = 38.
    # cov_weight = 1.
    cov_weight = 0.35
    full_cov = True
    q_sqrt = None
    jitter = 1e-4
    white = True
    fun_kwargs = {
        "X": gp.X,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.Y,
        "full_cov": full_cov,
        "q_sqrt": q_sqrt,
        "jitter": jitter,
        "white": white,
    }
    fun = single_mogpe_mixing_probability
    metric_tensor_fn_kwargs = {"fun": fun, "fun_kwargs": fun_kwargs}


class FakeSVGPMetricProb:
    from ProbGeo.mogpe import single_mogpe_mixing_probability

    gp = FakeSVGP()
    full_cov = True
    jitter = 1e-4
    white = True
    fun_kwargs = {
        "X": gp.Z,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.q_mu,
        "full_cov": full_cov,
        "q_sqrt": gp.q_sqrt,
        "jitter": jitter,
        "white": white,
    }
    fun = single_mogpe_mixing_probability
    metric_tensor_fn_kwargs = {"fun": fun, "fun_kwargs": fun_kwargs}


class FakeODESolver:
    num_timesteps = 100
    t_init = 0.0
    t_end = 1.0
    times = np.linspace(t_init, t_end, num_timesteps)
    int_method = "Radau"
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
    num_timesteps = 100000
    t_init = 0.0
    t_end = 1.0
    times = np.linspace(t_init, t_end, num_timesteps)
    int_method = "Radau"
    # int_method = 'RK45'
    root_tol = 0.0005
    maxfev = 1000
    pos_init = np.array([-1.5, -1.0])
    pos_init = np.array([-2.0, -0.25])
    # pos_init = np.array([-0.5, -0.7])
    pos_init = np.array([1.7, -0.7])
    pos_end_targ = np.array([1.8, 1.8])
    # vel_init_guess = np.array([-5, -3])
    vel_init_guess = np.array([500, 1000])
    state_init = np.concatenate([pos_init, vel_init_guess])


class FakeODESolverProb:
    num_timesteps = 1000
    t_init = 0.0
    t_end = 1.0
    times = np.linspace(t_init, t_end, num_timesteps)
    int_method = "Radau"
    # int_method = 'BDF'
    # int_method = 'RK45'
    root_tol = 0.0005
    a_tol = 0.05
    maxfev = 1000
    # pos_init = np.array([-1.5, -1.])
    pos_init = np.array([-1, -0.25])
    pos_end_targ = np.array([1.5, -2.5])
    # vel_init_guess = np.array([5.1, 2.1])
    # vel_init_guess = np.array([6.1, -10.1])
    vel_init_guess = np.array([0.5, -0.5])
    state_init = np.concatenate([pos_init, vel_init_guess])


class FakeODESolverProbSVGP:
    num_timesteps = 1000
    t_init = 0.0
    t_end = 1.0
    times = np.linspace(t_init, t_end, num_timesteps)
    int_method = "Radau"
    # int_method = 'BDF'
    # int_method = 'RK45'
    root_tol = 0.00005
    a_tol = 0.05
    maxfev = 1000
    # pos_init = np.array([-1.5, -1.])
    pos_init = np.array([-1, -0.25])
    pos_end_targ = np.array([1.8, 1.8])
    # vel_init_guess = np.array([5.1, 2.1])
    vel_init_guess = np.array([6.1, -10.1])
    state_init = np.concatenate([pos_init, vel_init_guess])


def test_shooting_geodesic_solver():
    from ProbGeo.metric_tensor import gp_metric_tensor

    solver = FakeODESolver()
    metric = FakeGPMetric()
    metric_fn = gp_metric_tensor
    opt_vel_init, geodesic_traj = shooting_geodesic_solver(
        solver.pos_init,
        solver.pos_end_targ,
        solver.vel_init_guess,
        metric_fn,
        metric.metric_fn_kwargs,
        solver.times,
        solver.t_init,
        solver.t_end,
        solver.int_method,
        solver.root_tol,
        solver.maxfev,
    )
    return opt_vel_init, geodesic_traj


def test_shooting_geodesic_solver_with_svgp():
    from ProbGeo.metric_tensor import gp_metric_tensor

    solver = FakeODESolverSVGP()
    metric = FakeSVGPMetric()
    metric_fn = gp_metric_tensor
    opt_vel_init, geodesic_traj = shooting_geodesic_solver(
        solver.pos_init,
        solver.pos_end_targ,
        solver.vel_init_guess,
        metric_fn,
        metric.metric_fn_kwargs,
        solver.times,
        solver.t_init,
        solver.t_end,
        solver.int_method,
        solver.root_tol,
        solver.maxfev,
    )
    return opt_vel_init, geodesic_traj


def test_shooting_geodesic_solver_with_prob():
    from ProbGeo.metric_tensor import metric_tensor_fn

    solver = FakeODESolverProb()
    metric = FakeGPMetricProb()
    metric_fn = metric_tensor_fn
    opt_vel_init, geodesic_traj = shooting_geodesic_solver(
        solver.pos_init,
        solver.pos_end_targ,
        solver.vel_init_guess,
        metric_fn,
        metric.metric_tensor_fn_kwargs,
        solver.times,
        solver.t_init,
        solver.t_end,
        solver.int_method,
        solver.root_tol,
        solver.maxfev,
    )
    return opt_vel_init, geodesic_traj


def test_shooting_geodesic_solver_with_prob_svgp():
    from ProbGeo.metric_tensor import metric_tensor_fn

    solver = FakeODESolverProbSVGP()
    metric = FakeSVGPMetricProb()
    metric_fn = metric_tensor_fn
    opt_vel_init, geodesic_traj = shooting_geodesic_solver(
        solver.pos_init,
        solver.pos_end_targ,
        solver.vel_init_guess,
        metric_fn,
        metric.metric_tensor_fn_kwargs,
        solver.times,
        solver.t_init,
        solver.t_end,
        solver.int_method,
        solver.root_tol,
        solver.maxfev,
    )
    return opt_vel_init, geodesic_traj


def test_and_plot_shooting_geodesic_solver():
    gp = FakeGP()
    solver = FakeODESolver()
    # plt.show()
    fig, axs = plot_gp_and_start_end(gp, solver)
    opt_vel_init, geodesic_traj = test_shooting_geodesic_solver()
    fig, axs = plot_traj(fig, axs, geodesic_traj.T)
    plt.show()


def test_and_plot_shooting_geodesic_solver_with_svgp():
    gp = FakeSVGP()
    solver = FakeODESolverSVGP()
    # plt.show()
    fig, axs = plot_svgp_and_start_end(gp, solver)
    opt_vel_init, geodesic_traj = test_shooting_geodesic_solver_with_svgp()
    fig, axs = plot_traj(fig, axs, geodesic_traj.T)
    plt.show()


def test_and_plot_shooting_geodesic_solver_with_prob():
    gp = FakeGP()
    solver = FakeODESolverProb()
    fig, ax = plot_mixing_prob_and_start_end(gp, solver)
    plt.show()
    fig, ax = plot_mixing_prob_and_start_end(gp, solver)
    opt_vel_init, geodesic_traj = test_shooting_geodesic_solver_with_prob()
    fig, axs = plot_traj(fig, ax, geodesic_traj.T)
    plt.show()


def test_and_plot_shooting_geodesic_solver_with_prob_svgp():
    import matplotlib.pyplot as plt
    from ProbGeo.mogpe import mogpe_mixing_probability
    from ProbGeo.visualisation.gp import plot_contourf
    from ProbGeo.visualisation.utils import create_grid

    # Plot manifold with start and end points
    gp = FakeSVGP()
    solver = FakeODESolverProbSVGP()

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mixing_probs = mogpe_mixing_probability(
        Xnew,
        gp.Z,
        gp.kernel,
        mean_func=gp.mean_func,
        f=gp.q_mu,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    fig, ax = plt.subplots(1, 1)
    plot_contourf(fig, ax, xx, yy, mixing_probs[:, 0:1])

    ax.scatter(solver.pos_init[0], solver.pos_init[1], marker="o", color="r")
    ax.scatter(
        solver.pos_end_targ[0], solver.pos_end_targ[1], color="r", marker="o"
    )
    ax.annotate("start", (solver.pos_init[0], solver.pos_init[1]))
    ax.annotate("end", (solver.pos_end_targ[0], solver.pos_end_targ[1]))
    # plt.show()

    (
        opt_vel_init,
        geodesic_traj,
    ) = test_shooting_geodesic_solver_with_prob_svgp()

    ax.scatter(geodesic_traj[0, :], geodesic_traj[1, :], marker="x", color="k")
    ax.plot(geodesic_traj[0, :], geodesic_traj[1, :], marker="x", color="k")
    plt.show()


class FakeMultipleShooting:
    num_timesteps = 1000
    # num_grid_points = 6
    num_grid_points = 10
    # num_grid_points = 10
    # num_grid_points = 3
    input_dim = 2
    t_init = 0.0
    t_end = 1.0
    times = np.linspace(
        t_init, t_end, num_timesteps * num_grid_points
    ).reshape(num_grid_points, num_timesteps)
    print(times.shape)
    pos_init = np.array([2.8, -2.5])
    pos_end_targ = np.array([-1.5, 2.8])
    # vel_init_guess = np.array([-5.641, 3.95])
    vel_init_guess = np.array([-0.5641, 0.395])
    # vel_init_guess = np.array([-1.5641, 1.395])
    # vel_init_guess = np.array([-1.5641, 1.395])
    # vel_init_guess = np.array([-2.5641, 3.395])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_grid_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_grid_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_grid_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)
    # state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)[0:1, :]


class FakeMultipleShootingSVGP:
    num_timesteps = 1000
    # num_grid_points = 6
    num_grid_points = 10
    # num_grid_points = 3
    input_dim = 2
    t_init = 0.0
    t_end = 1.0
    times = np.linspace(
        t_init, t_end, num_timesteps * num_grid_points
    ).reshape(num_grid_points, num_timesteps)
    pos_init = np.array([1.7, -0.7])
    pos_end_targ = np.array([1.8, 1.8])
    pos_end_targ = np.array([0.0, 1.8])
    # vel_init_guess = np.array([-5, -3])
    vel_init_guess = np.array([-0.05, -0.03])
    # vel_init_guess = np.array([500, 1000])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_grid_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_grid_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_grid_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)
    # state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)[0:1, :]


class FakeMultipleShootingProb:
    num_timesteps = 1000
    num_grid_points = 10
    input_dim = 2
    t_init = 0.0
    t_end = 1.0
    times = np.linspace(
        t_init, t_end, num_timesteps * num_grid_points
    ).reshape(num_grid_points, num_timesteps)
    pos_init = np.array([2.8, -2.5])
    pos_end_targ = np.array([-1.5, 2.8])
    vel_init_guess = np.array([-0.005641, 0.00395])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_grid_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_grid_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_grid_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


def test_multiple_shooting():
    from ProbGeo.metric_tensor import gp_metric_tensor
    from ProbGeo.solvers import multiple_shooting

    solver = FakeMultipleShooting()
    metric = FakeGPMetric()
    metric_fn = gp_metric_tensor
    geodesic_traj = multiple_shooting(
        solver.state_guesses, metric_fn, metric.metric_fn_kwargs, solver.times
    )
    return geodesic_traj


def test_multiple_shooting_svgp():
    from ProbGeo.metric_tensor import gp_metric_tensor
    from ProbGeo.solvers import multiple_shooting

    solver = FakeMultipleShootingSVGP()
    metric = FakeSVGPMetric()
    metric_fn = gp_metric_tensor
    geodesic_traj = multiple_shooting(
        solver.state_guesses, metric_fn, metric.metric_fn_kwargs, solver.times
    )
    return geodesic_traj


def test_multiple_shooting_prob():
    from ProbGeo.metric_tensor import metric_tensor_fn

    solver = FakeMultipleShootingProb()
    metric = FakeGPMetricProb()
    metric_fn = metric_tensor_fn
    from ProbGeo.solvers import multiple_shooting

    geodesic_traj = multiple_shooting(
        solver.state_guesses,
        metric_fn,
        metric.metric_tensor_fn_kwargs,
        solver.times,
    )
    return geodesic_traj


def test_and_plot_multiple_shooting():
    gp = FakeGP()
    solver = FakeMultipleShooting()
    fig, axs = plot_gp_and_start_end(gp, solver)
    # plt.show()
    geodesic_traj = test_multiple_shooting()
    fig, axs = plot_traj(fig, axs, geodesic_traj)
    plt.show()


def test_and_plot_multiple_shooting_svgp():
    gp = FakeSVGP()
    solver = FakeMultipleShootingSVGP()
    fig, axs = plot_svgp_and_start_end(gp, solver)
    plt.show()
    fig, axs = plot_svgp_and_start_end(gp, solver)
    geodesic_traj = test_multiple_shooting_svgp()
    fig, axs = plot_traj(fig, axs, geodesic_traj)
    plt.show()


def test_and_plot_multiple_shooting_prob():
    gp = FakeGP()
    solver = FakeMultipleShootingProb()
    fig, ax = plot_mixing_prob_and_start_end(gp, solver)
    plt.show()
    fig, ax = plot_mixing_prob_and_start_end(gp, solver)
    geodesic_traj = test_multiple_shooting_prob()
    fig, axs = plot_traj(fig, ax, geodesic_traj)
    plt.show()


if __name__ == "__main__":

    # test_and_plot_shooting_geodesic_solver()
    # test_and_plot_shooting_geodesic_solver_with_svgp()

    # with jax.disable_jit():
    # test_and_plot_shooting_geodesic_solver_with_prob()
    # test_and_plot_shooting_geodesic_solver_with_prob_svgp()

    # test_and_plot_shooting_geodesic_solver_with_prob()
    # test_and_plot_shooting_geodesic_solver_with_prob_svgp()

    # test_multiple_shooting()
    test_and_plot_multiple_shooting()
    # test_and_plot_multiple_shooting_svgp()
    # test_and_plot_multiple_shooting_prob()

    # jax.jit(test_multiple_shooting())
    # with jax.disable_jit():
    #     test_multiple_shooting()
