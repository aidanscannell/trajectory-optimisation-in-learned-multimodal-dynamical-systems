import os
import sys

import jax.numpy as np
import jax.scipy as sp
import matplotlib.pyplot as plt
from derivative_kernel_gpy import DiffRBF
from jax import jacfwd, jit, partial, vmap, jacrev
# from probabilistic_geodesic import value_and_jacfwd
# from matplotlib import cm
from utils.metric_utils import (create_grid, init_save_path, plot_gradient,
                                plot_mean_and_var, plot_mean_and_var_contour,
                                plot_metric_trace)

global jitter
jitter = 1e-4


def Kuu(inducing_inputs, kernel, jitter=1e-4):
    Kzz = kernel.K(inducing_inputs, inducing_inputs)
    Kzz += jitter * np.eye(len(inducing_inputs), dtype=Kzz.dtype)
    return Kzz


def Kuf(inducing_inputs, kernel, Xnew):
    return kernel.K(inducing_inputs, Xnew)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def single_sparse_gp_derivative_predict(x_star, X, z, q_mu, q_sqrt, kernel,
                                        mean_func, m_h_mu):
    def Kvar(x1, x2, z, q_mu, q_sqrt, kernel, mean_func):
        k_uu = Kuu(z, kernel)  # 200 200
        k_1u = Kuf(x1, kernel, z)  # 961 200
        k_u2 = Kuf(z, kernel, x2)  # 200 961
        k_ff = kernel.K(x1, x2)  # 961 961
        print('inside')
        print(k_uu.shape)
        print(k_1u.shape)
        print(k_u2.shape)
        print(k_ff.shape)
        # print(np.count_nonzero(np.isnan(k_uu)))
        # print(np.count_nonzero(np.isnan(k_1u)))
        # print(np.count_nonzero(np.isnan(k_u2)))
        # print(np.count_nonzero(np.isnan(k_ff)))

        Lu = sp.linalg.cholesky(k_uu, lower=True)
        A1 = sp.linalg.solve_triangular(Lu, k_1u.T, lower=True)
        A2 = sp.linalg.solve_triangular(Lu, k_u2, lower=True)
        ATA = A1.T @ A2

        Ls = np.squeeze(q_sqrt)

        # LLSLLST = A1.T @ LLS @ LLS.T @ A2
        LTA1 = Ls @ A1
        LTA2 = Ls @ A2
        print('LTA')
        print(LTA1.shape)
        print(LTA2.shape)

        cov = k_ff - ATA + LTA1.T @ LTA2
        # cov = k_ff + LTA1.T @ LTA2
        return cov

    def Kmean(x1, x2, z, q_mu, q_sqrt, kernel, mean_func):
        k_uu = Kuu(z, kernel)
        k_1u = Kuf(x1, kernel, z)
        k_ff = kernel.K(x1, x2)

        Lu = sp.linalg.cholesky(k_uu, lower=True)
        A1 = sp.linalg.solve_triangular(Lu, k_1u.T, lower=True)

        fmean = A1.T @ (q_mu - 0.)
        fmean += mean_func

        return fmean

    num_data = X.shape[0]
    x_star = x_star.reshape(-1, 2)

    # d2k = jacfwd(Kvar, (0, 1))(x_star, x_star, z, q_mu, q_sqrt, kernel,
    #                            mean_func)
    # d2k0 = np.squeeze(d2k[0])
    # d2k1 = np.squeeze(d2k[1])
    # d2k = np.array([d2k0, d2k1])
    print("d2k")
    # print(d2k.shape)
    d2k = jacrev(jacfwd(Kvar, (1)), (0))(x_star, x_star, z, q_mu, q_sqrt,
                                         kernel, mean_func)
    # d2k = hessian(Kvar_hess, 0)(x_star, kernel)
    # d2k = hessian(kernel.K)(x_star, x_star)
    print(d2k)
    d2k = np.squeeze(d2k)

    dkxs = jacfwd(Kvar, 1)(X, x_star, z, q_mu, q_sqrt, kernel, mean_func)
    print('dkxs')
    print(dkxs.shape)
    dkxs = dkxs.reshape(num_data, 2)
    print('dkxs')
    print(dkxs.shape)
    # dkxs += jitter * np.eye(dkxs.shape[1])

    print('HERERER')
    Kxx = Kvar(X, X, z, q_mu, q_sqrt, kernel, mean_func)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    A = sp.linalg.solve_triangular(chol, dkxs, lower=True)

    # TODO should both derivatives be wrt to x_star???

    mu_h = Kmean(X, X, z, q_mu, q_sqrt, kernel, mean_func)
    A2 = sp.linalg.solve_triangular(chol, mu_h, lower=True)

    # calculate mean and variance of J
    mu_j = A.T @ A2
    # print("mu_j")
    # print(mu_j.shape)
    # TODO does all of d2k need to be calculated for sparse GP
    # TODO should this be plus or minus
    cov_j = d2k - A.T @ A  # d2K doesn't need to be calculated
    cov_j = d2k - A.T @ A  # d2K doesn't need to be calculated
    # cov_j = -A.T @ A  # d2K doesn't need to be calculated
    cov_j = A.T @ A  # d2K doesn't need to be calculated
    print("cov_j")
    print(cov_j.shape)

    return mu_j, cov_j


def sparse_gp_derivative_predict(xy, X, z, q_mu, q_sqrt, kernel, mean_func,
                                 m_h_mu):
    """
    xy: test inputs to calulate predictive mean and variance at [M, num_data]
    X: training inputs [num_data, input_dim]
    Y: training outputs [num_data, output_dim]
    """
    input_dim = X.shape[1]
    print('xy.shape')
    print(xy.shape)
    print(input_dim)
    if xy.shape == (xy.shape[0], input_dim):
        mu_j, cov_j = vmap(single_sparse_gp_derivative_predict,
                           in_axes=(0, None, None, None, None, None, None,
                                    None))(xy, X, z, q_mu, q_sqrt, kernel,
                                           mean_func, m_h_mu)

    else:
        raise ValueError(
            'Test inputs array should be of shape (-1, input_dim)')
    return mu_j, cov_j


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def calc_G_map_sparse(c, X, z, q_mu, q_sqrt, kernel, mean_func, m_h_mu):
    c = c.reshape(1, -1)
    input_dim = X.shape[1]
    print('q_mu.shape')
    print(q_mu.shape)
    output_dim = q_mu.shape[1]
    print('output_dim')
    print(output_dim)
    mu_j, cov_j = single_sparse_gp_derivative_predict(c, X, z, q_mu, q_sqrt,
                                                      kernel, mean_func,
                                                      m_h_mu)

    # mu_j = np.nan_to_num(mu_j)
    # cov_j = np.nan_to_num(cov_j)

    mu_jT = mu_j.T
    # assert mu_jT.shape == (1, input_dim)
    print('here')
    print(mu_j.shape)
    print(mu_jT.shape)

    jTj = np.matmul(mu_j, mu_jT)  # [input_dim x input_dim]
    assert jTj.shape == (input_dim, input_dim)
    var_weight = 100.
    # var_weight = 0.1
    G = jTj + var_weight * output_dim * cov_j  # [input_dim x input_dim]
    assert G.shape == (input_dim, input_dim)
    return G, mu_j, cov_j


def gp_predict(x_star, X, Y, kernel, mean_func=0., jitter=1e-4):
    num_data = X.shape[0]

    Kxx = kernel.K(X, X)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    assert chol.shape == (num_data, num_data)
    kinvy = sp.linalg.solve_triangular(
        chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))
    assert kinvy.shape == (num_data, 1)

    # calculate mean and variance of J
    Kxs = kernel.K(X, x_star)
    mu = np.dot(Kxs.T, kinvy)

    Kss = kernel.K(x_star, x_star)
    v = sp.linalg.solve_triangular(chol, Kxs, lower=True)
    vT = v.T
    cov = Kss - np.matmul(vT, v)
    mu = mu + mean_func
    return mu, cov


def gp_predict_sparse(x_star, z, mean_func, q_mu, q_sqrt, kernel, jitter=1e-8):
    Kmm = Kuu(z, kernel)
    Kmn = Kuf(z, kernel, x_star)
    Knn = kernel.K(x_star, x_star)
    Lm = sp.linalg.cholesky(Kmm, lower=True)
    A = sp.linalg.solve_triangular(Lm, Kmn, lower=True)

    fmean = A.T @ q_mu
    fmean = fmean + mean_func

    # fvar = Knn - np.sum(np.square(A))
    fvar = Knn - A.T @ A
    q_sqrt = np.squeeze(q_sqrt)
    LTA = q_sqrt @ A
    fvar = fvar + LTA.T @ LTA

    return fmean, fvar


def gp_predict_sparse_sym(x1,
                          x2,
                          z,
                          mean_func,
                          q_mu,
                          q_sqrt,
                          kernel,
                          jitter=1e-8):
    k_uu = Kuu(z, kernel)
    k_1u = Kuf(x1, kernel, z)
    k_u2 = Kuf(z, kernel, x2)
    k_ff = kernel.K(x1, x2)

    Lu = sp.linalg.cholesky(k_uu, lower=True)
    A1 = sp.linalg.solve_triangular(Lu, k_1u.T, lower=True)
    A2 = sp.linalg.solve_triangular(Lu, k_u2, lower=True)
    ATA = A1.T @ A2

    Ls = np.squeeze(q_sqrt)

    LTA1 = Ls @ A1
    LTA2 = Ls @ A2

    fvar = k_ff - ATA + LTA1.T @ LTA2

    # print('here1234')
    # print(A1.shape)
    # Amu = sp.linalg.solve_triangular(Lu, q_mu, lower=True)
    # print(Amu.shape)
    # fmean = A1.T @ Amu
    fmean = A1.T @ (q_mu - 0.)
    fmean += mean_func

    return fmean, fvar


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
    print('here')
    print(lengthscale)

    kernel = DiffRBF(X.shape[1],
                     variance=var,
                     lengthscale=lengthscale,
                     ARD=True)
    return X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var


if __name__ == "__main__":
    X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var = load_data_and_init_kernel(
        # filename='saved_models/27-2/137/params_from_model.npz')
        filename='./saved_models/model-fake-data/1210/params_from_model.npz')
    # filename='saved_models/26-2/1247/params_from_model.npz')
    # filename='saved_models/20-2/189/params_from_model.npz')

    save_path = init_save_path()

    mu_j_sparse, cov_j_sparse = sparse_gp_derivative_predict(
        xy, X, z, q_mu, q_sqrt, kernel, mean_func, m_h_mu)

    axs = plot_mean_and_var_contour(xx, yy, h_mu.reshape(xx.shape),
                                    h_var.reshape(xx.shape))
    plt.suptitle('Original GP - predicted using BMNSVGP')
    axs = plot_mean_and_var(X, m_h_mu, m_h_var)
    plt.suptitle('Original GP - predicted using BMNSVGP')
    plt.savefig(save_path + 'original_gp_from_model.pdf', transparent=True)

    # mu, var = gp_predict(xy, X, m_h_mu, kernel, mean_func=mean_func)
    # var = np.diag(var).reshape(-1, 1)
    # axs = plot_mean_and_var(xy, mu, var)
    # plt.suptitle("Full GP prediction func")

    # mu, var = gp_predict_sparse(xy, z, mean_func, q_mu, q_sqrt, kernel)
    # var = np.diag(var).reshape(-1, 1)
    # axs_sparse = plot_mean_and_var(xy, mu, var)
    # plt.scatter(z[:, 0], z[:, 1])
    # plt.suptitle("Sparse GP prediction func")
    # axs = plot_mean_and_var_contour(xx, yy, mu.reshape(xx.shape),
    #                                 var.reshape(xx.shape))
    # plt.suptitle("Sparse GP prediction func")

    # mu_sparse, var_sparse = gp_predict_sparse_LTA(xy, z, mean_func, q_mu,
    #                                               q_sqrt, kernel)
    print(xy.shape)
    mu_sparse, var_sparse = gp_predict_sparse_sym(xy, xy, z, mean_func, q_mu,
                                                  q_sqrt, kernel)
    var_sparse = np.diag(var_sparse).reshape(-1, 1)
    axs = plot_mean_and_var_contour(xx, yy, mu_sparse.reshape(xx.shape),
                                    var_sparse.reshape(xx.shape))
    plt.suptitle("Sparse GP prediction func (symmetric K(x,x'))")

    mu_sparse, var_sparse = gp_predict_sparse(xy, z, mean_func, q_mu, q_sqrt,
                                              kernel)
    var_sparse = np.diag(var_sparse).reshape(-1, 1)
    axs = plot_mean_and_var(xy, mu_sparse, var_sparse)
    plt.suptitle("Sparse GP prediction func with LTA.T LTA")
    plt.savefig(save_path + 'original_gp_predicted.pdf', transparent=True)
    # plt.show()

    print('Calculating trace of metric, cov_j and mu_j...')
    G, mu_j, cov_j = vmap(calc_G_map_sparse,
                          in_axes=(0, None, None, None, None, None, None,
                                   None))(xy, X, z, q_mu, q_sqrt, kernel,
                                          mean_func, m_h_mu)
    print('Done calculating metric')

    var_j = vmap(np.diag, in_axes=(0))(cov_j)

    axs = plot_gradient(xy, mu_j, var_j, mu_sparse, var_sparse, save_path)
    # plt.show()

    axs = plot_metric_trace(xy, G, mu_sparse, var_sparse, save_path)

    plt.show()
