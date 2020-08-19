import jax.numpy as np
import jax.scipy as sp
import matplotlib.pyplot as plt
from derivative_kernel_gpy import DiffRBF
from jax import jit, partial, vmap, jacfwd, hessian, jacrev
from utils.metric_utils import (create_grid, init_save_path,
                                load_data_and_init_kernel_fake, plot_gradient,
                                plot_mean_and_var, plot_metric_trace)
from jax.config import config
config.update("jax_debug_nans", True)

global jitter
jitter = 1e-4


@partial(jit, static_argnums=(1, 2, 3))
def single_gp_derivative_predict(xy, X, Y, kernel):
    """
    xy: a single test input [1, input_dim]
    """
    xy = xy.reshape(1, -1)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    Kxx = kernel.K(X, X)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    # TODO check cholesky is implemented correctly
    # kinvy = sp.linalg.solve_triangular(
    #     chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))
    kinvy = sp.linalg.solve_triangular(chol, Y, lower=True)

    # dk_dt0 = kernel.dK_dX(xy, X, 0)
    # dk_dt1 = kernel.dK_dX(xy, X, 1)
    # dk_dtT = np.stack([dk_dt0, dk_dt1], axis=1)
    # dk_dtT = np.squeeze(dk_dtT)

    def Kvar(x1, x2, kernel):
        fvar = kernel.K(x1, x2)
        return fvar

    dk_dt = jacfwd(Kvar, 0)(xy, X, kernel)
    # print('dk_dt')
    # print(dk_dt.shape)
    dk_dt = dk_dt.reshape(-1, input_dim)
    dk_dtT = dk_dt.T

    v = sp.linalg.solve_triangular(chol, dk_dtT.T, lower=True)

    l2 = kernel.lengthscale**2
    l2 = np.diag(l2)
    d2k_dtt = -l2 * kernel.K(xy, xy)

    # calculate mean and variance of J
    # mu_j = np.dot(dk_dtT, kinvy)
    mu_j = v.T @ kinvy
    cov_j = d2k_dtt - np.matmul(v.T, v)  # d2Kd2t doesn't need to be calculated
    return mu_j, cov_j


@partial(jit, static_argnums=(1, 2, 3))
def single_gp_derivative_predict_pred(x_star, X, Y, kernel):
    """
    xy: a single test input [1, input_dim]
    """
    x_star = x_star.reshape(1, -1)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    def Kvar(x1, x2, kernel):
        fvar = kernel.K(x1, x2)
        return fvar

    num_data = X.shape[0]
    Kxx = Kvar(X, X, kernel)
    print('Kxx.shape')
    print(Kxx.shape)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)

    # # TODO check cholesky is implemented correctly
    # print(chol.shape)
    # print(Y.shape)
    # kinvy = sp.linalg.solve_triangular(
    #     chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))

    def Kvar_hess(x, kernel):
        fvar = kernel.K(x, x)
        print('asfasdaffsd')
        print(fvar.shape)
        # fvar = np.squeeze(fvar)
        # print(fvar.shape)
        return fvar

    print('before')
    # d2k = jacfwd(jacrev(Kvar))
    # d2k = j2k(x_star, x_star, kernel)
    d2k = jacrev(jacfwd(Kvar, (1)), (0))(x_star, x_star, kernel)
    # d2k = hessian(Kvar_hess, 0)(x_star, kernel)
    # d2k = hessian(kernel.K)(x_star, x_star)
    print(d2k)
    d2k = np.squeeze(d2k)

    # # d2k = jacfwd(Kvar, (0, 1))(x_star, x_star, kernel)
    # x_star1 = np.array([0.2, 0.1]).reshape(1, 2)
    # print(x_star1.shape)
    # # d2k = hessian(Kvar_hess, (0))(x_star1, kernel)
    # d2k = hessian(Kvar_hess, (0))(x_star, kernel)
    # print(d2k)
    # # d2k0 = d2k[0][0]
    # d2k0 = np.squeeze(d2k[0])
    # print(d2k0)
    # # d2k1 = d2k[1][1]
    # # d2k1 = np.squeeze(d2k[1][1])
    # # print(d2k1)
    # # d2k = np.array([d2k0, d2k1])
    # d2k = d2k0
    print("d2k")
    print(d2k.shape)

    dksx = jacfwd(Kvar, 0)(x_star, X, kernel)
    print('dksx')
    print(dksx.shape)
    dksx = dksx.reshape(num_data, 2)
    print('dksx')
    print(dksx.shape)
    # dkxs += jitter * np.eye(dkxs.shape[1])

    # dkxs = jacfwd(Kvar, 1)(X, x_star, kernel)
    # print('dkxs')
    # print(dkxs.shape)
    # dkxs = dkxs.reshape(num_data, 2)
    # # dkxs = dkxs.reshape(2, num_data)
    # print('dkxs')
    # print(dkxs.shape)

    A1 = sp.linalg.solve_triangular(chol, dksx, lower=True)
    print('A1')
    print(A1.shape)

    # calculate mean and variance of J
    # print("kinvy")
    # print(kinvy.shape)
    # mu_j = np.dot(dk_dtT, kinvy)
    # v = sp.linalg.solve_triangular(chol, dk_dtT.T, lower=True)
    print('ASssssss')
    print(A1.shape)
    A2 = sp.linalg.solve_triangular(chol, Y, lower=True)
    print(A2.shape)
    mu_j = A1.T @ A2
    print(mu_j.shape)

    # calculate mean and variance of J
    # mu_j = np.dot(dk_dtT, kinvy)
    # cov_j = d2k - np.matmul(v.T, v)  # d2Kd2t doesn't need to be calculated
    ATA = A1.T @ A1
    print('ATA')
    print(ATA.shape)
    cov_j = d2k - ATA
    cov_j = d2k
    # cov_j = -ATA
    return mu_j, cov_j


def gp_derivative_predict(xy, X, Y, kernel):
    """
    xy: test inputs to calulate predictive mean and variance at [M, num_data]
    X: training inputs [num_data, input_dim]
    Y: training outputs [num_data, output_dim]
    """
    input_dim = X.shape[1]
    print('xy.shape')
    print(xy.shape)
    print(input_dim)
    if xy.shape == (961, input_dim):
        mu_j, cov_j = vmap(single_gp_derivative_predict,
                           in_axes=(0, None, None, None))(xy, X, Y, kernel)
    else:
        raise ValueError(
            'Test inputs array should be of shape (-1, input_dim)')
    return mu_j, cov_j


@partial(jit, static_argnums=(1, 2, 3))
def calc_G_map(c, X, Y, kernel):
    c = c.reshape(1, -1)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    print('calc_G_map')
    mu_j, cov_j = single_gp_derivative_predict(c, X, Y, kernel)
    # mu_j, cov_j = single_gp_derivative_predict_pred(c, X, Y, kernel)

    # print("number NaNs in mu_j")
    # print(np.count_nonzero(np.isnan(mu_j)))
    # print("number NaNs in cov_j")
    # print(np.count_nonzero(np.isnan(cov_j)))
    # mu_j = np.nan_to_num(mu_j)
    # cov_j = np.nan_to_num(cov_j)

    mu_jT = mu_j.T
    # assert mu_jT.shape == (1, input_dim)

    jTj = np.matmul(mu_j, mu_jT)  # [input_dim x input_dim]
    # jTj = np.matmul(mu_jT, mu_j)  # [input_dim x input_dim]
    assert jTj.shape == (input_dim, input_dim)
    var_weight = 0.1
    var_weight = 0.35
    G = jTj + var_weight * output_dim * cov_j  # [input_dim x input_dim]
    assert G.shape == (input_dim, input_dim)
    return G, mu_j, cov_j


def gp_predict(x_star, X, Y, kernel, mean_func=0., jitter=1e-4):
    num_data = X.shape[0]
    # input_dim = X.shape[1]

    Kxx = kernel.K(X, X)
    # print(Kxx.shape)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    assert chol.shape == (num_data, num_data)
    kinvy = sp.linalg.solve_triangular(
        chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))
    assert kinvy.shape == (num_data, 1)

    # calculate mean and variance of J
    Kxs = kernel.K(X, x_star)
    # print(x_star.shape)
    # print(Kxs.shape)
    mu = np.dot(Kxs.T, kinvy)
    # print(mu_j.shape)
    # assert mu_j.shape == (input_dim, 1)

    Kss = kernel.K(x_star, x_star)
    v = sp.linalg.solve_triangular(chol, Kxs, lower=True)
    # assert v.shape == (num_data, input_dim)
    vT = v.T
    cov = Kss - np.matmul(vT, v)
    # assert cov_j.shape == (input_dim, input_dim)
    # cov *= 100
    mu = mu + mean_func
    return mu, cov


if __name__ == "__main__":

    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
        filename='./saved_models/params_fake.npz')
    Y = a_mu

    save_name = init_save_path(dir_name="visualise_metric")

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
