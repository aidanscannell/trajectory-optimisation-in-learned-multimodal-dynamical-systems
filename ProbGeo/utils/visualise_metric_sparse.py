import os
import sys

import jax.numpy as np
import matplotlib.pyplot as plt
from jax import vmap
from visualise_metric import plot_gradient, plot_metric_trace, create_grid, plot_mean_and_var

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from derivative_kernel_gpy import DiffRBF
from sparse_probabilistic_metric import (calc_G_map_sparse,
                                         gp_predict_sparse_sym,
                                         gp_predict_sparse_LTA)
from probabilistic_metric import calc_G_map, gp_predict


def load_data_and_init_kernel_sparse(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscales = params['l']  # [2]
    kern_vars = params['kern_var']  # [1]
    lik_vars = params['lik_var']  # [1]
    X = params['x']  # [num_data x 2]
    Y = params['y']  # [num_data x 2]
    zs = params['z_f']  # [num_data x 2]
    q_mus = params['q_mu_f']  # [num_data x 1] mean of alpha
    q_sqrts = params['q_sqrts_f']  # [num_data x 1] variance of alpha
    mean_func = params['mean_func_f']

    print('inssss')
    print(kern_vars.shape)
    print(lengthscales.shape)
    print(lik_vars.shape)
    kernels = []
    for mode in range(0, kern_vars.shape[0]):
        kernel = DiffRBF(X.shape[1],
                         variance=kern_vars[mode, ...],
                         lengthscale=lengthscales[mode, ...],
                         ARD=True)
        kernels.append(kernel)
        print(kernel)
    print(kernels)
    return X, Y, q_mus, q_sqrts, zs, kernels, mean_func


# def mixture_of_gps(mixtures, zs, kernels, mean_funcs, q_mus, q_sqrts):
#     for weighting, z, kernel, mean_func, q_mu, q_sqrt in zip(mixtures, zs, kernels, mean_funcs, q_mus, s_sqrts):
#     gp_predict_sparse_LTA()


def load_data_and_init_kernel(filename):
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


def load_data_and_init_kernel_fake_sparse(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscale = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    z = params['z']  # [num_data x 2]
    q_mu = params['q_mu']  # [num_data x 1] mean of alpha
    q_sqrt = params['q_sqrt']  # [num_data x 1] variance of alpha
    mean_func = params['mean_func']

    kernel = DiffRBF(X.shape[1],
                     variance=var,
                     lengthscale=lengthscale,
                     ARD=True)
    return X, z, q_mu, q_sqrt, kernel, mean_func


if __name__ == "__main__":

    X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var = load_data_and_init_kernel(
        # filename='../saved_models/28-2/1036/params_from_model.npz')
        filename='../saved_models/model-fake-data/1210/params_from_model.npz')
    # filename='../saved_models/3-3/166/params_from_model.npz')
    # # filename='../saved_models/27-2/137/params_from_model.npz')
    # Y = m_h_mu
    # X, z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_fake_sparse(
    #     # filename='../saved_models/params_fake_sparse.npz')
    #     filename='../saved_models/27-2/137/params_from_model.npz')

    # X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
    #     filename='../saved_models/18-3/1637/params_from_model_f.npz')

    import datetime
    from pathlib import Path
    date = datetime.datetime.now()
    date_str = str(date.day) + "-" + str(date.month) + "/" + str(
        date.time()) + "/"
    save_name = "/Users/aidanscannell/Developer/python-projects/BMNSVGP/images/visualise_metric_sparse/" + date_str
    Path(save_name).mkdir(parents=True, exist_ok=True)

    # var = np.diag(cov).reshape(-1, 1)
    print(m_h_mu.shape)
    print(m_h_var.shape)
    print(xy.shape)
    axs = plot_mean_and_var(X, m_h_mu, m_h_var)
    plt.suptitle('Original GP - predicted using BMNSVGP')

    # plot original GP
    xy, xx, yy = create_grid(X, N=961)

    # mu, cov = gp_predict(xy, X, a_mu, kernel)
    # var = np.diag(cov).reshape(-1, 1)
    # axs = plot_mean_and_var(xy, mu, var)
    # plt.suptitle('Original GP')
    # plt.savefig(save_name + 'original_gp.pdf', transparent=True)

    # mu, cov = gp_predict(xy, X, m_h_mu, kernel)
    mu_sparse, cov_sparse = gp_predict_sparse_LTA(xy, z, mean_func, q_mu,
                                                  q_sqrt, kernel)
    # var = np.diag(cov).reshape(-1, 1)
    var_sparse = np.diag(cov_sparse).reshape(-1, 1)
    # axs = plot_mean_and_var(xy, mu, var)
    # plt.suptitle('Original GP none sparse')
    axs_sparse = plot_mean_and_var(xy, mu_sparse, var_sparse)
    plt.suptitle('Original GP sparse')
    plt.savefig(save_name + 'original_gp.pdf', transparent=True)

    mu_sparse_sym, cov_sparse_sym = gp_predict_sparse_sym(
        xy, X, z, mean_func, q_mu, q_sqrt, kernel)
    var_sparse_sym = np.diag(cov_sparse_sym).reshape(-1, 1)
    axs = plot_mean_and_var(xy, mu_sparse_sym, var_sparse_sym)
    plt.suptitle('Original GP sparse sym')

    # mu, cov = gp_predict_sparse_LTA(xy, z, mean_func, q_mu, q_sqrt, kernel)
    # var = np.diag(cov).reshape(-1, 1)
    # axs = plot_mean_and_var(xy, mu, var)
    # plt.suptitle('Original GP')
    # plt.savefig(save_name + 'original_gp.pdf', transparent=True)

    # print('Calculating trace of metric, cov_j and mu_j...')
    # G, mu_j, cov_j = vmap(calc_G_map,
    #                       in_axes=(0, None, None, None))(xy, X, m_h_mu, kernel)
    # print('Done calculating metric')

    # var_j = vmap(np.diag, in_axes=(0))(cov_j)

    # axs = plot_gradient(xy, mu_j, var_j, mu_sparse, var_sparse, save_name)

    # axs = lot_metric_trace(xy, G, mu_sparse, var_sparse, save_name)
    plt.show()

    print('Calculating trace of metric, cov_j and mu_j...')
    G, mu_j, cov_j = vmap(calc_G_map_sparse,
                          in_axes=(0, None, None, None, None, None, None,
                                   None))(xy, X, z, q_mu, q_sqrt, kernel,
                                          mean_func, m_h_mu)
    print('Done calculating metric')

    var_j = vmap(np.diag, in_axes=(0))(cov_j)

    axs = plot_gradient(xy, mu_j, var_j, mu_sparse, var_sparse, save_name)

    axs = plot_metric_trace(xy, G, mu_sparse, var_sparse, save_name)

    plt.show()
