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


class ProbabilisticMetricTensor:
    def __init__(self,
                 X,
                 var_fun,
                 var_args,
                 mean_func,
                 var_weight=1,
                 jitter=1e-4):
        '''
        This class calculates a probabilistic Riemannian metric tensor on a manifold represented by GPs.
        Assuming we have a GP h,
            p(h | X, ...) = N(m_h, S_h(X, X))
        We have a dist over its Jacobian,
            p(J | h, X, ...) = N(m_j, S_j)
            m_j = dS_h(x_*,x)/dx_* S_h(x,x)^{-1} m_h
            S_j = d2S_h(x_*, x_*)/d(x_*x_*) - dS_h(x_*,x)/dx_* S_h(x,x)^{-1} dS_h(x,x_*)/dx_*
        Expected Riemannian metric tensor,
            E_{p(J|h,X,...)}[G(x)] = E[J^T J] = E[J]^T E[J] + output_dim * S_j
        Calculate the mean and variance of the Jacobian for a test input.
            X:           Inputs [num_data, input_dim].
            var_fun:     S_h - Function that calculates the prior covariance of GP (allows switching between
                         sparse and standard. It should have signature var_fun(x_star, X, *var_args).
            var_args:    Arguments to pass into var_fun after x_star and X.
            mean_func:   m_h - Constant value for GP mean function.
                         TODO: extend to non-constant mean functions
            var_weight:  Weighting of covariance term in expected metric tensor calculation.
        '''
        self.output_dim = 1
        self.X = X
        self.num_data = X.shape[0]
        self.input_dim = X.shape[1]
        self.var_fun = var_fun
        self.var_args = var_args
        self.mean_func = mean_func
        self.var_weight = var_weight
        self.jitter = jitter

        Kxx = var_fun(X, X, *var_args)
        Kxx += self.jitter * np.eye(Kxx.shape[0])
        self.chol_var_fun_xx = sp.linalg.cholesky(Kxx, lower=True)

    def _single_gp_derivative_predict(self, x_star):
        '''
        Calculate the mean and variance of the Jacobian for a test input.
            x_star:      test input [1, input_dim]
        '''
        x_star = x_star.reshape(1, self.input_dim)

        # calculate hessian with two derivative observations
        # TODO does all of d2k need to be calculated for sparse GP
        d2k = jacrev(jacfwd(self.var_fun, (1)), (0))(x_star, x_star,
                                                     *self.var_args)
        d2k = np.squeeze(d2k)

        # calculate derivative wrt one derivative observation
        dkxs = jacfwd(var_fun, 1)(self.X, x_star, *self.var_args)
        dkxs = dkxs.reshape(self.num_data, self.input_dim)

        A = sp.linalg.solve_triangular(self.chol_var_fun_xx, dkxs, lower=True)
        A2 = sp.linalg.solve_triangular(self.chol_var_fun_xx,
                                        self.mean_func,
                                        lower=True)

        # calculate mean and variance of J
        mu_j = A.T @ A2
        cov_j = d2k - A.T @ A  # d2K doesn't need to be calculated
        cov_j = -A.T @ A  # d2K doesn't need to be calculated
        # cov_j = A.T @ A  # d2K doesn't need to be calculated
        # TODO: select correct covariance
        return mu_j, cov_j

    def calc_expected_metric_tensor(self, test_inputs):
        '''
        Calculates the expected Riemannian metric tensor,
            E_{p(J|X,h,x_*)}[G(x)] = E[J]^T E[J] + ouput_dim * S_j
            with Riemannian metric, a G(x) b = a^T J^T J b,
            where J is the Jacobian.
        test_inputs:   {x_*}^num_tests - test inputs [num_tests, input_dim]
        '''
        def calc_expected_metric_tensor_single(x_star):
            mu_j, cov_j = self._single_gp_derivative_predict(x_star)
            assert mu_j.shape == (self.input_dim, 1)
            assert cov_j.shape == (self.input_dim, self.input_dim)

            expected_jTj = np.matmul(mu_j, mu_j.T)  # [input_dim x input_dim]
            assert expected_jTj.shape == (self.input_dim, self.input_dim)

            expected_metric_tensor = expected_jTj + self.var_weight * self.output_dim * cov_j  # [input_dim x input_dim]
            assert expected_metric_tensor.shape == (self.input_dim,
                                                    self.input_dim)
            return expected_metric_tensor, mu_j, cov_j

        if len(test_inputs.shape) == 2:
            if test_inputs.shape != (
                    self.input_dim,
                    self.input_dim) and test_inputs.shape[0] == self.input_dim:
                print('Transposing test inputs from shape: ',
                      test_inputs.shape)
                test_inputs = test_inputs.T
        elif len(test_inputs.shape) == 1:
            if test_inputs.shape[0] == self.input_dim:
                test_inputs = test_inputs.reshape(1, self.input_dim)
            else:
                raise ValueError(
                    'Test inputs should have a dimension equal to self.input_dim'
                )
        else:
            raise ValueError(
                'Test inputs should be of shape (num_test_inputs, input_dim) or (input_dim, ) or (input_dim, num_test_inputs)'
            )
        self.num_test_inputs = test_inputs.shape[0]
        assert test_inputs.shape == (self.num_test_inputs, self.input_dim)

        self.expected_metric_tensor, self.mu_j, self.cov_j = vmap(
            calc_expected_metric_tensor_single, in_axes=(0))(test_inputs)
        return self.expected_metric_tensor


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
    sparse = False
    if sparse:
        var_weight = 100
        X, var_fun, var_args, mean_func, mu, var, save_path, test_inputs = test_sparse_gp(
        )
    else:
        var_weight = 0.1
        X, var_fun, var_args, mean_func, mu, var, save_path, test_inputs = test_standard_gp(
        )

    axs = plot_mean_and_var(test_inputs, mu, var)
    plt.suptitle('Original GP')
    plt.savefig(save_path + 'original_gp.pdf', transparent=True)

    print('Initialising metric tensor...')
    metric_tensor = ProbabilisticMetricTensor(X, var_fun, var_args, mean_func,
                                              var_weight)
    print('Done initialising metric tensor')

    print('Calculating expected metric tensor...')
    expected_metric_tensor = metric_tensor.calc_expected_metric_tensor(
        test_inputs)
    print('Done calculating expected metric tensor')

    var_j = vmap(np.diag, in_axes=(0))(metric_tensor.cov_j)

    axs = plot_gradient(test_inputs, metric_tensor.mu_j, var_j, mu, var,
                        save_path)
    # plt.show()

    axs = plot_metric_trace(test_inputs, expected_metric_tensor, mu, var,
                            save_path)

    plt.show()
