import os
import re
import sys

import jax.numpy as np
import matplotlib.pyplot as plt
from jax import vmap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from derivative_kernel_gpy import DiffRBF
from probabilistic_metric import (calc_G_map, gp_predict)


def plot_mean_and_var(X, Y_mean, Y_var, llabel='mean', rlabel='variance'):
    fig, axs = plt.subplots(1, 2, figsize=(24, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    surf_mean = plot_contour(X, Y_mean, ax=axs[0])
    surf_var = plot_contour(X, Y_var, ax=axs[1])
    cbar = fig.colorbar(surf_mean, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label(llabel)
    cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label(rlabel)
    return axs


def plot_contour(X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    cont = ax.tricontourf(X[:, 0], X[:, 1], Y.reshape(-1), 15)
    return cont


def create_grid(X, N):
    x1_low, x1_high, x2_low, x2_high = X[:, 0].min(), X[:, 0].max(
    ), X[:, 1].min(), X[:, 1].max()
    # x1_low, x1_high, x2_low, x2_high = -2., 3., -3., 3.
    sqrtN = int(np.sqrt(N))
    xx = np.linspace(x1_low, x1_high, sqrtN)
    yy = np.linspace(x2_low, x2_high, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    return xy, xx, yy


def plot_gradient(xy,
                  mu_j,
                  var_j,
                  mu,
                  var,
                  save_name='../../images/visualise_metric/'):
    axs_j = plot_mean_and_var(xy, mu_j[:, 0, 0], mu_j[:, 1, 0], '$dim_1$',
                              '$dim_2$')
    plt.suptitle("$\mu_j$")
    plt.savefig(save_name + 'gradient_mean.pdf', transparent=True)

    axs = plot_mean_and_var(xy, mu, var)
    for ax in axs:
        ax.quiver(xy[:, 0], xy[:, 1], mu_j[:, 0, 0], mu_j[:, 1, 0])
    plt.suptitle("$\mu_j$")
    plt.savefig(save_name + 'gradient_mean_quiver.pdf', transparent=True)

    axs_j = plot_mean_and_var(xy, var_j[:, 0], var_j[:, 1], '$dim_1$',
                              '$dim_2$')
    plt.suptitle("$\Sigma_j$")
    plt.savefig(save_name + 'gradient_variance.pdf', transparent=True)

    axs = plot_mean_and_var(xy, mu, var)
    for ax in axs:
        ax.quiver(xy[:, 0], xy[:, 1], var_j[:, 0], var_j[:, 1])
    plt.suptitle("$\Sigma_j$")
    plt.savefig(save_name + 'gradient_variance_quiver.pdf', transparent=True)
    return axs


def plot_metric_trace(xy,
                      G,
                      mu,
                      var,
                      save_name='../../images/visualise_metric/'):
    traceG = np.trace(G, axis1=1, axis2=2)

    plot_contour(xy, traceG)
    plt.title('Tr(G(x))')
    plt.savefig(save_name + 'trace(G(x)).pdf', transparent=True)

    axs = plot_mean_and_var(xy, mu, var)
    plt.suptitle("G(x)")
    for ax in axs:
        ax.quiver(xy[:, 0], xy[:, 1], G[:, 0, 0], G[:, 1, 1])
    plt.savefig(save_name + 'G(x).pdf', transparent=True)
    return axs


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

    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
        filename='../saved_models/params_fake.npz')
    Y = a_mu

    import datetime
    from pathlib import Path
    date = datetime.datetime.now()
    date_str = str(date.day) + "-" + str(date.month) + "/" + str(
        date.time()) + "/"
    save_name = "../../images/visualise_metric/" + date_str
    Path(save_name).mkdir(parents=True, exist_ok=True)

    # plot original GP
    xy, xx, yy = create_grid(X, N=961)
    mu, cov = gp_predict(xy, X, a_mu, kernel)
    var = np.diag(cov).reshape(-1, 1)
    axs = plot_mean_and_var(xy, mu, var)
    plt.suptitle('Original GP')
    plt.savefig(save_name + 'original_gp.pdf', transparent=True)

    print('Calculating trace of metric, cov_j and mu_j...')
    G, mu_j, cov_j = vmap(calc_G_map, in_axes=(0, None, None, None))(xy, X, Y,
                                                                     kernel)
    print('Done calculating metric')
    # mu_j, var_j = gp_derivative_predict(xy, X, Y, kernel)
    var_j = vmap(np.diag, in_axes=(0))(cov_j)

    axs = plot_gradient(xy, mu_j, var_j, mu, var, save_name)

    axs = plot_metric_trace(xy, G, mu, var, save_name)

    plt.show()
