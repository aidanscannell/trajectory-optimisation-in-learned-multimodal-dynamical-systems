import jax.numpy as np
import jax.scipy as sp
from jax import jit, partial, vmap

from derivative_kernel_gpy import DiffRBF

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
    kinvy = sp.linalg.solve_triangular(
        chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))

    dk_dt0 = kernel.dK_dX(xy, X, 0)
    dk_dt1 = kernel.dK_dX(xy, X, 1)
    dk_dtT = np.stack([dk_dt0, dk_dt1], axis=1)
    dk_dtT = np.squeeze(dk_dtT)

    v = sp.linalg.solve_triangular(chol, dk_dtT.T, lower=True)

    l2 = kernel.lengthscale**2
    l2 = np.diag(l2)
    d2k_dtt = -l2 * kernel.K(xy, xy)

    # calculate mean and variance of J
    mu_j = np.dot(dk_dtT, kinvy)
    cov_j = d2k_dtt - np.matmul(v.T, v)  # d2Kd2t doesn't need to be calculated
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
    mu_j, cov_j = single_gp_derivative_predict(c, X, Y, kernel)

    mu_jT = mu_j.T
    # assert mu_jT.shape == (1, input_dim)

    jTj = np.matmul(mu_j, mu_jT)  # [input_dim x input_dim]
    assert jTj.shape == (input_dim, input_dim)
    var_weight = 0.1
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


# if __name__ == "__main__":

#     # X, Y, kernel = load_data_and_init_kernel(
#     #     filename='saved_models/params.npz')
#     X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
#         filename='saved_models/params_fake.npz')
#     Y = a_mu

#     # plot original GP
#     xy, xx, yy = create_grid(X, N=961)
#     mu, var = gp_predict(xy, X, a_mu, kernel)
#     var = np.diag(var).reshape(-1, 1)
#     axs = plot_mean_and_var(xy, mu, var)
#     plt.suptitle('Original GP')

#     xy, xx, yy = create_grid(X, N=961)
#     axs = plot_gradeint(xy, X, Y, kernel)
#     axs = plot_metric_trace(xy, X, Y, kernel)

#     plt.show()
