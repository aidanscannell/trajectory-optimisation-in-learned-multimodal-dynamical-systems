from functools import reduce

import jax.numpy as np
import jax.scipy as scipy
from jax import jacfwd, jit, partial

from derivative_kernel_gpy import DiffRBF
from value_and_jac import value_and_jacfwd

global jitter
jitter = 1e-4


@partial(jit, static_argnums=(1, 2, 3))
def calc_G_map(c, X, Y, kernel):
    c = c.reshape(1, -1)
    # num_data = X.shape[0]
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    # assert kinvy.shape == (num_data, 1)

    Kxx = kernel.K(X, X)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = scipy.linalg.cholesky(Kxx, lower=True)
    # assert chol.shape == (num_data, num_data)
    # TODO check cholesky is implemented correctly
    kinvy = scipy.linalg.solve_triangular(
        chol.T, scipy.linalg.solve_triangular(chol, Y, lower=True))

    dk_dt0 = kernel.dK_dX(c, X, 0)
    # assert dk_dt0.shape == (c.shape[0], num_data)
    dk_dt1 = kernel.dK_dX(c, X, 1)
    # assert dk_dt1.shape == (c.shape[0], num_data)
    dk_dtT = np.stack([dk_dt0, dk_dt1], axis=1)
    dk_dtT = np.squeeze(dk_dtT)
    # assert dk_dtT.shape == (input_dim, num_data)
    # assert dk_dt.shape == (num_data, input_dim)

    # calculate mean and variance of J
    mu_j = np.dot(dk_dtT, kinvy)
    # assert mu_j.shape == (input_dim, 1)
    # mu = np.dot(cross_cov.T, kinvy) + ymean
    v = scipy.linalg.solve_triangular(chol, dk_dtT.T, lower=True)
    # v = scipy.linalg.solve_triangular(dk_dtT, chol, lower=True)
    # var = (amp * cov_map(exp_quadratic, xtest) - np.dot(v.T, v))
    # assert v.shape == (num_data, input_dim)
    # TODO should this be negative and is it correct
    # prod_lengthscales = 1 / reduce(lambda x, y: x * y, kernel.lengthscale, 1)
    # d2k_dtt = -np.eye(input_dim) * kernel.K(c, c)

    l2 = kernel.lengthscale**2
    l2 = np.diag(l2)
    d2k_dtt = -l2 * kernel.K(c, c)
    # diag_d2k_dtt = prod_lengthscales * kernel.K(c, c)
    # d2k_dtt = -np.eye(input_dim) * diag_d2k_dtt
    # d2k_dtt = 0.
    # print('vTv')
    # print(np.matmul(v.T, v).shape)
    # print(d2k_dtt)
    cov_j = d2k_dtt - np.matmul(v.T, v)  # d2Kd2t doesn't need to be calculated
    # assert cov_j.shape == (input_dim, input_dim)

    mu_jT = mu_j.T
    # assert mu_jT.shape == (1, input_dim)

    jTj = np.matmul(mu_j, mu_jT)  # [input_dim x input_dim]
    # jTj = np.matmul(mu_jT, mu_j)  # [1 x 1]
    print(jTj.shape)
    # assert jTj.shape == (input_dim, input_dim)
    var_weight = 1.
    G = jTj + var_weight * output_dim * cov_j  # [input_dim x input_dim]
    # assert G.shape == (input_dim, input_dim)
    return G, jTj, cov_j


@partial(jit, static_argnums=(1, 2, 3))
def calc_vecG(c, X, Y, kernel):
    global G
    G, _, _ = calc_G_map(c, X, Y, kernel)
    input_dim = X.shape[1]
    vecG = G.reshape(input_dim * input_dim, )  # order doesnt matter as diag
    return vecG


@partial(jit, static_argnums=(2, 3, 4))
def geodesic_fun(c, g, X, Y, kernel):
    if len(c.shape) < 2:
        c = c.reshape(1, c.shape[0])
    if len(g.shape) < 2:
        g = g.reshape(1, g.shape[0])
    kronC = np.kron(g, g).T

    # this works with my new JAX function
    val_grad_func = value_and_jacfwd(calc_vecG, 0)
    vecG, dvecGdc = val_grad_func(c, X, Y, kernel)
    G = vecG.reshape(c.shape[1], c.shape[1])

    # this works with normal JAX but is slower as it requires to calls to calc_G_map
    # grad_vecG = jacfwd(calc_vecG, 0)
    # dvecGdc = grad_vecG(c, X, Y, kernel)
    # G, _, _ = calc_G_map(c, X, Y, kernel)

    invG = np.linalg.inv(G)
    dvecGdc = dvecGdc[:, 0, :].T
    return -0.5 * invG @ dvecGdc @ kronC


def load_data_and_init_kernel(filename='saved_models/params.npz'):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscale = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    y = params['a']  # [num_data x 2] meen and var of alpha
    Y = y[0:1, :, 0].T  # [num_data x 1]
    kernel = DiffRBF(2, variance=var, lengthscale=lengthscale, ARD=True)
    return X, Y, kernel


# def setup_global_vars(x=None, y=None, filename='saved_models/params.npz'):
#     global kinvy, kernel, chol, X, Y
#     params = np.load(filename)
#     lengthscale = params['l']  # [2]
#     var = params['var']  # [1]
#     kernel = DiffRBF(2, variance=var, lengthscale=lengthscale, ARD=True)
#     if x is None:
#         X = params['x']  # [num_data x 2]
#     else:
#         X = x
#     if y is None:
#         y = params['a']  # [num_data x 2] meen and var of alpha
#         Y = y[0:1, :, 0].T  # [num_data x 1]
#     else:
#         Y = y

#     # X, Y, kernel = load_data_and_init_kernel()

#     Kxx = kernel.K(X, X)
#     Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
#     chol = scipy.linalg.cholesky(Kxx, lower=True)
#     # assert chol.shape == (num_data, num_data)
#     # TODO check cholesky is implemented correctly
#     kinvy = scipy.linalg.solve_triangular(
#         chol.T, scipy.linalg.solve_triangular(chol, Y, lower=True))

if __name__ == "__main__":

    m = 10
    x1 = np.linspace(0, 1, m)
    x2 = np.linspace(0, 1, m)
    dx1 = np.linspace(0.1, 0.1, m)
    dx2 = np.linspace(0.1, 0.1, m)
    y = np.vstack([x1, x2, dx1, dx2])
    c = np.vstack((y[0], y[1])).T
    g = np.vstack((y[2], y[3])).T

    # dydt = vmap(geodesic_fun, in_axes=(0))(c, g)
    # print(c.shape)
    # print(g.shape)
    # setup_global_vars()
    X, Y, kernel = load_data_and_init_kernel(
        filename='saved_models/params.npz')

    # import timeit
    from time import process_time
    # tic = process_time()
    # timeit.timeit(dydt=geodesic_fun_eff(c[0:1, :], g[0:1, :]))
    # toc = process_time()
    # print(toc - tic)
    # print(dydt)

    tic = process_time()
    dydt = geodesic_fun(c[0:1, :], g[0:1, :], X, Y, kernel)
    toc = process_time()
    print(toc - tic)
    print(dydt)

    tic = process_time()
    dydt = geodesic_fun(c[0:1, :], g[0:1, :], X, Y, kernel)
    toc = process_time()
    print(toc - tic)
    print(dydt)

    tic = process_time()
    dydt = geodesic_fun(c[0:1, :], g[0:1, :], X, Y, kernel)
    toc = process_time()
    print(toc - tic)
    print(dydt)
