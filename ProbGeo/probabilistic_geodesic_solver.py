import os
import sys

import jax.numpy as np
import jax.scipy as sp
import matplotlib.pyplot as plt
from derivative_kernel_gpy import DiffRBF
from jax import jacfwd, jit, partial, vmap, jacrev
from utils.metric_utils import (create_grid, init_save_path, plot_gradient,
                                plot_mean_and_var, plot_mean_and_var_contour,
                                plot_metric_trace)
from utils.value_and_jac import value_and_jacfwd

class ProbabilisticGeodesicSolver:
    def __init__(self, metric_tensor, )
    self.metric_tensor = metric_tensor
    self.input_dim = 2

    # @partial(jit, static_argnums=(1, 2, 3))
    @partial(jit)
    # TODO: what is dimension of test_inputs here?
    def calc_vec_expected_metric_tensor(self, test_inputs):
    # def calc_vec_expected_metric_tensor(c):
        # G, _, _ = calc_G_map(c, X, Y, kernel)
        expected_metric_tensor = self.metric_tensor.calc_expected_metric_tensor(test_inputs)
        # order of reshape doesnt matter as diag
        vec_exp_metric_tensor = expected_metric_tensor.reshape(self.input_dim * self.input_dim, )
        return vec_exp_metric_tensor


    @partial(jit)
    def geodesic_fun(self, c, g):
        if len(c.shape) < 2:
            c = c.reshape(1, c.shape[0])
        if len(g.shape) < 2:
            g = g.reshape(1, g.shape[0])
        kronC = np.kron(g, g).T

        def calc_vec_expected_metric_tensor(test_inputs):
            expected_metric_tensor = self.metric_tensor.calc_expected_metric_tensor(test_inputs)
            vec_exp_metric_tensor = expected_metric_tensor.reshape(self.input_dim * self.input_dim, )
            return vec_exp_metric_tensor

        # this works with my new JAX function
        vec_exp_metric_tensor, dvec_dc = value_and_jacfwd(calc_vec_expected_metric_tensor, 0)(c)
        # vecG, dvecGdc = val_grad_func(c, X, Y, kernel)
        exp_metric_tensor = vec_exp_metric_tensor.reshape(self.input_dim, self.input_dim)

        # TODO implement cholesky inversion
        inv_exp_metric_tensor = np.linalg.inv(expected_metric_tensor)
        dvecGdc = dvecGdc[:, 0, :].T
        return -0.5 * inv_exp_metric_tensor @ dvec_dc @ kronC
        # return -0.5 * invG @ dvecGdc @ kronC

    def compute_zprime(x, z, ode_func, ode_args):
        """
        Compute the value of the vector z's derivative at a point given the value of the vector function z and the
        independent variable x. The form of this calculation is specified by the vector ODE. Return a vector for the
        derivative.

        :param x: indpendent variable, the domain of the problem is x=0 to L
        :param z: 2-vec of the variable representing system of equations, z = [y, y']
        :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
        :return: 2-vec of values for z' = f(x, z)
        """
        z_0 = np.array([z[0], z[1]])
        zprime_0 = np.array([z[2], z[3]])
        zprime_1 = ode_func(z_0, zprime_0, *ode_args)
        return [z[2], z[3], zprime_1[0, 0], zprime_1[1, 0]]


    def integrate_over_domain(z_at_0,
                            ode_func,
                            ode_args,
                            integrator='RK45',
                            length=1,
                            step=0.2,
                            max_step=0.001,
                            silent=True):
        """
        Call runge-kutta repeatedly to integrate the vector function z over the full domain (specified by length). Return
        a list of 2-vecs which are the value of the vector z at every point in the domain discretized by 'step'. Note that
        runge-kutta object calls x as "t" and z as "y".

        :param z_at_0: the value of the vector z=[y, y'] at the left boundary point. should be list or array.
        :param integrator: a string for the integration method to use e.g. 'RK45', 'DOP853'
        :param length: the length of the domain to integrate on
        :param step: the step size with which to discretize the domain
        :param silent: bool indicating whether to print the progress of the integrator
        :return: array of 2-vecs - the value of vector z obtained by integration for each point in the discretized domain.
        """
        t = np.linspace(0., length, int(length / step))
        z_at_0 = np.array(z_at_0)

        integrator = solve_ivp(
            fun=lambda t, y: compute_zprime(t, y, ode_func, ode_args),
            t_span=(0., length),
            y0=z_at_0,
            t_eval=t,
            # max_step=max_step,
            method=integrator)
        # dense_output=True)
        print(integrator)
        return integrator.t, integrator.y


    def solve_bvp_tj(y_at_0,
                    y_at_length,
                    yprime_at_0_guess,
                    ode_func,
                    ode_args,
                    integrator,
                    length=1,
                    step=0.2,
                    root_tol=0.05,
                    maxfev=10000,
                    max_step=0.001,
                    silent=True):
        """
        Numerically find the value for y'(0) that gives us a propagated
        (integrated) solution matching most closely with other known boundary
        condition y(L) which is proportional junction temperature.

        :param y_at_0: the known boundary condition y(0)
        :param y_at_length: the known boundary condition y(L)
        :param yprime_at_0_guess: initial guess of y'(0)
        :param length: the length of the domain to integrate on
        :param step: the step size with which to discretize the domain
        :param silent: bool indicating whether to print the progress of the integrator
        :param root_tol: tolerance of root finder
        :return: the optimized estimate of y' at the left boundary point giving the most accurate integrated solution.
        """
        def residuals(yprime_at_0, y_at_0, y_at_length):
            # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
            print('Trying yprime_at_0: (%.8f, %.8f)' %
                (yprime_at_0[0], yprime_at_0[1]))
            z_at_0 = [y_at_0[0], y_at_0[1], yprime_at_0[0], yprime_at_0[1]]
            xs, zs = integrate_over_domain(z_at_0,
                                        ode_func,
                                        ode_args,
                                        integrator,
                                        length=length,
                                        step=step,
                                        max_step=max_step,
                                        silent=silent)
            # y_at_length_integrated = np.array(zs)[-1, 0:2]
            y_at_length_integrated = np.array(zs)[0:2, -1]

            # Return the difference between y(L) found by numerical integrator and the true value
            y_at_length = np.array(y_at_length)
            error = y_at_length - y_at_length_integrated
            print('Residual [y(0)-y(L)] for current y\'(0): ', error)
            return error

        yprime_at_0_guess = np.array(yprime_at_0_guess)

        rt = root(
            residuals,
            yprime_at_0_guess,
            args=(y_at_0, y_at_length),
            # jac=loss_jac,
            options={'maxfev': maxfev},
            tol=root_tol)
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


def test_sparse_gp(
    save_img_dir='visualise_sparse_metric_tensor',
    filename='./saved_models/model-fake-data/1210/params_from_model.npz'):

    from utils.sparse_gp_helpers import gp_predict_sparse, gp_predict_sparse_sym, Kmean_sparse, Kvar_sparse

    def load_data_and_init_kernel(filename):
        print('Loading data and kernel hyperparameters...')
        # Load kernel hyper-params and create kernel
        params = np.load(filename)
        lengthscale = params['l']  # [2]
        var = params['var']  # [1]
        X = params['x']  # [num_data x 2]
        Y = params['y']  # [num_data x 2]
        z = params['z']  # [num_data x 2]
        q_mu = params['q_mu']  # [num_data x 1] mean of alpha
        q_sqrt = params['q_sqrt']  # [num_data x 1] variance of alpha
        h_mu = params['h_mu']  # [num_data x 1] mean of alpha
        h_var = params['h_var']  # [num_data x 1] variance of alpha
        m_h_mu = params['m_h_mu']  # [num_data x 1] mean of alpha
        m_h_var = params['m_h_var']  # [num_data x 1] variance of alpha
        xx = params['xx']
        yy = params['yy']
        xy = params['xy']
        mean_func = params['mean_func']

        kernel = DiffRBF(X.shape[1],
                         variance=var,
                         lengthscale=lengthscale,
                         ARD=True)
        return X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var

    X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func_h, xx, yy, xy, m_h_mu, m_h_var = load_data_and_init_kernel(
        filename=filename)
    save_path = init_save_path(save_img_dir)
    test_inputs = xy

    # calculate original GP for plotting
    mu_sparse, cov_sparse = gp_predict_sparse_sym(test_inputs, test_inputs, z,
                                                  mean_func_h, q_mu, q_sqrt,
                                                  kernel)
    var_sparse = np.diag(cov_sparse).reshape(-1, 1)

    mean_func = Kmean_sparse(X, X, z, q_mu, kernel,
                             mean_func_h)  # [num_data, 1]

    var_fun = Kvar_sparse
    var_args = [z, q_sqrt, kernel]
    return X, var_fun, var_args, mean_func, mu_sparse, var_sparse, save_path, test_inputs


if __name__ == "__main__":

    sparse = True
    # sparse = False
    if sparse:
        var_weight = 100
        X, var_fun, var_args, mean_func, mu, var, save_path, test_inputs = test_sparse_gp(
        )
    else:
        var_weight = 0.1
        X, var_fun, var_args, mean_func, mu, var, save_path, test_inputs = test_standard_gp(
        )

    print('Initialising metric tensor...')
    metric_tensor = ProbabilisticMetricTensor(X, var_fun, var_args, mean_func,
                                              var_weight)
    print('Done initialising metric tensor')

    geodesic_solver = ProbabilisticGeodesicSolver(metric_tensor)

    # print('Calculating expected metric tensor...')
    # expected_metric_tensor = metric_tensor.calc_expected_metric_tensor(
    #     test_inputs)
    # print('Done calculating expected metric tensor')

    # var_j = vmap(np.diag, in_axes=(0))(metric_tensor.cov_j)

    # axs = plot_gradient(test_inputs, metric_tensor.mu_j, var_j, mu, var,
    #                     save_path)
    # # plt.show()

    # axs = plot_metric_trace(test_inputs, expected_metric_tensor, mu, var,
    #                         save_path)

    # plt.show()
