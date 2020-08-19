from jax import vmap
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

from utils.metric_utils import (create_grid, init_save_path, plot_gradient,
                                plot_mean_and_var, plot_mean_and_var_contour,
                                plot_metric_trace)
# from utils.visualise_metric import load_data_and_init_kernel_fake, create_grid, plot_mean_and_var
from utils.sparse_gp_helpers import gp_predict_sparse
from utils.gp_helpers import gp_predict
# from probabilistic_geodesic import geodesic_fun
from derivative_kernel_gpy import DiffRBF

from jax import numpy as np


class GeodesicSolver:
    def __init__(self,
                 ode_func,
                 ode_args=None,
                 integrator='RK45',
                 length=1,
                 step=0.2,
                 max_step=0.001,
                 root_tol=0.05,
                 maxfev=10000,
                 silent=True):
        self.ode_func = ode_func
        self.ode_args = ode_args
        self.integrator = integrator
        self.length = length
        self.max_step = max_step
        self.root_tol = root_tol
        self.maxfev = maxfev
        self.silent = silent

        self.t = np.linspace(0., length, int(length / step))

    def _compute_zprime(self, x, z):
        z_0 = np.array([z[0], z[1]])
        zprime_0 = np.array([z[2], z[3]])
        zprime_1 = self.ode_func(z_0, zprime_0, *self.ode_args)
        return [z[2], z[3], zprime_1[0, 0], zprime_1[1, 0]]

    def integrate_over_domain(self, z_at_0):
        z_at_0 = np.array(z_at_0)

        integrator = solve_ivp(
            # fun=lambda t, y: self.compute_zprime(t, y, ode_func, ode_args),
            fun=self._compute_zprime,
            t_span=(0., self.length),
            y0=z_at_0,
            t_eval=self.t,
            # max_step=max_step,
            method=self.integrator)
        # dense_output=True)
        # print(integrator)
        return integrator.t, integrator.y

    def residuals(self, yprime_at_0, y_at_0, y_at_length):
        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        print('Trying yprime_at_0: (%.8f, %.8f)' %
              (yprime_at_0[0], yprime_at_0[1]))
        z_at_0 = [y_at_0[0], y_at_0[1], yprime_at_0[0], yprime_at_0[1]]
        xs, zs = self.integrate_over_domain(z_at_0)
        # y_at_length_integrated = np.array(zs)[-1, 0:2]
        y_at_length_integrated = np.array(zs)[0:2, -1]

        # Return the difference between y(L) found by numerical integrator and the true value
        y_at_length = np.array(y_at_length)
        error = y_at_length - y_at_length_integrated
        print('Residual [y(0)-y(L)] for current y\'(0): ', error)
        return error

    def solve_bvp_tj(self, y_at_0, y_at_length, yprime_at_0_guess):
        print('inside solve')
        yprime_at_0_guess = np.array(yprime_at_0_guess)

        rt = root(
            self.residuals,
            yprime_at_0_guess,
            args=(y_at_0, y_at_length),
            # jac=loss_jac,
            options={'maxfev': self.maxfev},
            tol=self.root_tol)
        print('after root')
        yprime_at_0_estimate = rt.x
        return yprime_at_0_estimate


def test_standard_gp(save_img_dir='visualise_metric_tensor',
                     filename='./saved_models/params_fake.npz'):

    from probabilistic_metric import gp_predict

    def load_data_and_init_kernel_fake(filename):
        # Load kernel hyper-params and create kernel
        params = np.load(filename)
        l = params['l']  # [2]
        var = params['var']  # [1]
        X = params['x']  # [num_data x 2]
        a_mu = params['a_mu']  # [num_data x 1] mean of alpha
        a_var = params['a_var']  # [num_data x 1] variance of alpha
        kernel = DiffRBF(X.shape[1], variance=var, lengthscale=l, ARD=True)
        return X, a_mu, a_var, kernel

    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(filename)
    Y = a_mu
    save_path = init_save_path(save_img_dir)

    # plot original GP
    xy, xx, yy = create_grid(X, N=961)
    mu, cov = gp_predict(xy, X, a_mu, kernel)
    var = np.diag(cov).reshape(-1, 1)
    test_inputs = xy

    mean_func = a_mu
    var_fun = kernel.K
    var_args = []
    return X, var_fun, var_args, mean_func, mu, var, save_path, test_inputs


def load_data_and_init_kernel_fake(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    l = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    a_mu = params['a_mu']  # [num_data x 1] mean of alpha
    a_var = params['a_var']  # [num_data x 1] variance of alpha
    kernel = DiffRBF(X.shape[1], variance=var, lengthscale=l, ARD=True)
    return X, a_mu, a_var, kernel


if __name__ == "__main__":
    # Set parameters for integrator
    length = 1.
    step = 0.01
    max_step = None
    integrator = 'RK45'
    integrator = 'LSODA'
    integrator = 'Radau'
    # ode_func = geodesic_fun
    # ode_func = geodesic_fun_sparse

    # Set parameters for root finder
    root_tol = 0.005
    maxfev = None  # max function evaluations for root finder
    maxfev = 1000  # max function evaluations for root finder

    # Set boundary conditions and create state vector at t=0
    yprime_at_0_guess = [-5.64100643, 3.95933075]  # for fake alpha
    y_at_length = [-0.5, 1.8]
    y_at_length = [-1.5, 2.8]
    y_at_0 = [2., -2.2]
    z_at_0 = y_at_0 + yprime_at_0_guess

    # Load probabilistic manifold (GP) data
    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    Y = a_mu
    # X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var = load_data_and_init_kernel(
    #     # filename='saved_models/27-2/137/params_from_model.npz')
    #     filename='./saved_models/model-fake-data/1210/params_from_model.npz')
    # Y = m_h_mu
    # mean_func += 2.

    # Initialise args for ode func (compute_zprime)
    from probabilistic_geodesic import geodesic_fun
    ode_func = geodesic_fun
    ode_args = (X, Y, kernel)
    # ode_args = (X, z, q_mu, q_sqrt, kernel, mean_func, m_h_mu)

    geodesic_solver = GeodesicSolver(ode_func,
                                     ode_args=ode_args,
                                     integrator=integrator,
                                     length=length,
                                     step=step,
                                     max_step=max_step,
                                     root_tol=root_tol,
                                     maxfev=maxfev)

    # Plot manifold with start and end points
    xy, xx, yy = create_grid(X, N=961)
    test_inputs = xy
    mu, cov = gp_predict(xy, X, Y, kernel)
    var = np.diag(cov).reshape(-1, 1)
    axs = plot_mean_and_var(xy, mu, var)
    for ax in axs:
        # ax.plot(x_alpha, y_alpha, 'b')
        # ax.scatter(X[:, 0], X[:, 1], color='k', marker='x')
        ax.scatter(y_at_0[0], y_at_0[1], marker='o', color='r')
        ax.scatter(y_at_length[0], y_at_length[1], color='r', marker='o')
        ax.annotate("start", (y_at_0[0], y_at_0[1]))
        ax.annotate("end", (y_at_length[0], y_at_length[1]))

    # plt.show()

    yprime_at_0_estimate = geodesic_solver.solve_bvp_tj(
        y_at_0, y_at_length, yprime_at_0_guess)

    yprime_at_0_estimate = [yprime_at_0_estimate[0], yprime_at_0_estimate[1]]
    z_at_0 = y_at_0 + yprime_at_0_estimate
    xs, zs = geodesic_solver.integrate_over_domain(z_at_0)

    plt.close()
    axs = plot_mean_and_var(xy, mu, var)
    x = np.array(y_at_0).reshape(1, 2)
    z = np.array(zs).T
    z = np.append(x, z[:, 0:2], axis=0)
    for ax in axs:
        ax.scatter(z[:, 0], z[:, 1], marker='x', color='k')
        ax.plot(z[:, 0], z[:, 1], marker='x', color='k')
        ax.scatter(y_at_0[0], y_at_0[1], marker='o', color='r')
        ax.scatter(y_at_length[0], y_at_length[1], color='r', marker='o')
        ax.annotate("start", (y_at_0[0], y_at_0[1]))
        ax.annotate("end", (y_at_length[0], y_at_length[1]))
    plt.show()
