import jax
import matplotlib.pyplot as plt

from jax import numpy as np
from ProbGeo.ode import geodesic_ode


class FakeGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    cov_weight = 1.
    X, Y, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    cov_fn = kernel.K
    mean_func = 0.
    q_sqrt = None


class FakeSVGPQuad:
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse, load_data_and_init_kernel_mogpe
    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../models/saved_models/params_from_model.npz')
    Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_mogpe(
        save_model_dir=
        # "../models/saved_models/quadcopter/09-10-115323-param_dict.pickle")
        # "../models/saved_models/quadcopter/09-08-153416-param_dict.pickle")
        # "../models/saved_models/quadcopter/09-08-144846-param_dict.pickle")
        "../models/saved_models/quadcopter/10-02-104545-param_dict.pickle")


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


class FakeGPMetric:
    gp = FakeGP()
    # cov_weight = 38.
    # cov_weight = 1.
    # cov_weight = 0.35
    cov_weight = 2.
    # cov_weight = 0.01
    # cov_weight = 0.3
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


class FakeSVGPMetricQuad:
    gp = FakeSVGPQuad()
    # cov_weight = 38.
    # cov_weight = 50.
    # cov_weight = 500.
    cov_weight = 0.3
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
    # from ProbGeo.metric_tensor import gp_metric_tensor
    # fun = gp_metric_tensor


class FakeCollocationSVGPQuad:
    # num_col_points = 20
    num_col_points = 100
    input_dim = 2
    # t_init = 0.
    t_init = -1.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
    # pos_init = np.array([-1.7, -0.7])
    # pos_end_targ = np.array([0.1, 1.8])
    pos_init = np.array([-1.4, -1.])
    # pos_mid = np.array([-1., 1.5])
    # pos_mid = np.array([-.5, 1.])
    pos_mid = np.array([-.5, 0.5])
    # pos_mid = np.array([-1.5, 1.5])
    # pos_mid = np.array([-1., 1.])
    pos_end_targ = np.array([1.5, 1.9])
    pos11_guesses = np.linspace(pos_init[0], pos_mid[0],
                                int(num_col_points / 2))
    pos21_guesses = np.linspace(pos_init[1], pos_mid[1],
                                int(num_col_points / 2))
    pos12_guesses = np.linspace(pos_mid[0], pos_end_targ[0],
                                int(num_col_points / 2))
    pos22_guesses = np.linspace(pos_mid[1], pos_end_targ[1],
                                int(num_col_points / 2))
    pos1_guesses = np.concatenate([pos11_guesses, pos12_guesses])
    pos2_guesses = np.concatenate([pos21_guesses, pos22_guesses])
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)

    # vel_init_guess = np.array([0.05, 0.03])
    # vel_init_guess = np.array([0.15, 0.13])
    # vel_init_guess = np.array([10.15, 10.13])
    vel_init_guess = np.array([0.005, 0.003])
    # pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    # pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    # pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


# TODO test sparse and non sparse cov_fn
def test_geodesic_ode():
    from ProbGeo.metric_tensor import gp_metric_tensor
    ode = FakeODE()
    metric_fn = gp_metric_tensor
    state_prime = geodesic_ode(ode.t_init, ode.state_init, metric_fn,
                               ode.metric_fn_args)


def test_geodesic_ode_svgp_quad():
    from ProbGeo.metric_tensor import gp_metric_tensor
    ode = FakeCollocationSVGPQuad()
    metric_fn = gp_metric_tensor
    state_prime = geodesic_ode(ode.t_init, ode.state_init, metric_fn,
                               ode.metric_fn_args)


def plot_ode_gp():
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict
    gp = FakeGP()
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(Xnew,
                         gp.X,
                         kernel=gp.kernel,
                         mean_func=gp.mean_func,
                         f=gp.Y,
                         q_sqrt=gp.q_sqrt,
                         full_cov=False)
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    from ProbGeo.metric_tensor import gp_metric_tensor
    ode = FakeODE()
    metric = FakeGPMetric()
    metric_fn = gp_metric_tensor

    def ode_fn(state_init):
        print('inside ode')
        print(state_init.shape)
        print(ode.vel_init_guess.shape)
        state_init = np.concatenate([state_init, ode.vel_init_guess])
        print('after concat')
        print(state_init.shape)
        state_prime = geodesic_ode(ode.times, state_init, metric_fn,
                                   metric.metric_fn_kwargs)
        return state_prime

    state_primes = jax.vmap(ode_fn)(Xnew)
    print('state primes')
    print(state_primes.shape)
    print(state_primes)

    for ax in axs:
        ax.quiver(Xnew[:, 0], Xnew[:, 1], state_primes[:, 2], state_primes[:,
                                                                           3])
    #     fig, ax = plot_start_and_end_pos(fig, ax, solver)
    # return fig, axs
    plt.show()


def plot_ode_svgp_quad():
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict
    gp = FakeSVGPQuad
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(Xnew,
                         gp.Z,
                         kernel=gp.kernel,
                         mean_func=gp.mean_func,
                         f=gp.q_mu,
                         q_sqrt=gp.q_sqrt,
                         full_cov=True)
    # full_cov=False)
    var = np.diag(var)
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    from ProbGeo.metric_tensor import gp_metric_tensor
    ode = FakeCollocationSVGPQuad()
    metric = FakeSVGPMetricQuad()
    metric_fn = gp_metric_tensor

    def ode_fn(state_init):
        print('inside ode')
        print(state_init.shape)
        print(ode.vel_init_guess.shape)
        state_init = np.concatenate([state_init, ode.vel_init_guess])
        print('after concat')
        print(state_init.shape)
        state_prime = geodesic_ode(ode.times, state_init, metric_fn,
                                   metric.metric_fn_kwargs)
        return state_prime

    state_primes = jax.vmap(ode_fn)(Xnew)
    print('state primes')
    print(state_primes.shape)
    print(state_primes)

    for ax in axs:
        # ax.quiver(Xnew[:, 0], Xnew[:, 1], state_primes[:, 0], state_primes[:,
        #                                                                    1])
        ax.quiver(Xnew[:, 0], Xnew[:, 1], state_primes[:, 2], state_primes[:,
                                                                           3])
    #     fig, ax = plot_start_and_end_pos(fig, ax, solver)
    # return fig, axs
    plt.show()


if __name__ == "__main__":

    # test_geodesic_ode()

    # plot_ode_gp()
    plot_ode_svgp_quad()
