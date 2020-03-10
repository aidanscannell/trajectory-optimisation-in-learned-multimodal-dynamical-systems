import jax.numpy as np
import jax.scipy as sp
from jax import jit, partial, vmap, jacfwd, hessian
# import numpy as np
# from utils.visualise_metric import create_grid, plot_mean_and_var
import matplotlib.pyplot as plt
from matplotlib import cm
# import gpflow as gpf

from derivative_kernel_gpy import DiffRBF
from probabilistic_geodesic import value_and_jacfwd

global jitter
jitter = 1e-4


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def single_sparse_gp_derivative_predict(x_star, X, z, q_mu, q_sqrt, kernel,
                                        mean_func, m_h_mu):
    def Ksparse(x1, x2, z, q_mu, q_sqrt, kernel, mean_func):
        Kmm = Kuu(z, kernel)
        Kmn = Kuf(z, kernel, x2)
        Knm = Kuf(x1, kernel, z)
        Knn = kernel.K(x1, x2)
        Lm = sp.linalg.cholesky(Kmm, lower=True)
        A = sp.linalg.solve_triangular(Lm, Kmn, lower=True)
        # A = sp.linalg.solve_triangular(Lm, A, lower=True)

        fvar = Knn - A.T @ A

        q_sqrt = np.squeeze(q_sqrt)
        LTA = q_sqrt @ A
        fvar = fvar + LTA.T @ LTA
        return fvar

    num_data = X.shape[0]
    # TODO add mean_func to q_mu

    x_star = x_star.reshape(-1, 2)

    d2k = jacfwd(Ksparse, (0, 1))(x_star, x_star, z, q_mu, q_sqrt, kernel,
                                  mean_func)
    d2k0 = np.squeeze(d2k[0])
    d2k1 = np.squeeze(d2k[1])
    d2k = np.array([d2k0, d2k1])
    print("d2k")
    print(d2k.shape)

    Kxx = Ksparse(X, X, z, q_mu, q_sqrt, kernel, mean_func)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    print("kxx")
    print(Kxx.shape)

    Kxs = Ksparse(X, x_star, z, q_mu, q_sqrt, kernel, mean_func)
    Kxs = Kxs + jitter * np.eye(Kxs.shape[0])
    print("kxs")
    print(Kxs.shape)

    chol = sp.linalg.cholesky(Kxx, lower=True)
    print("chol")
    print(chol.shape)
    # print('q_mu.shape')
    # print(q_mu.shape)

    # calculate mean of GP (h)
    Kmm = Kuu(z, kernel)
    Kmm = Kmm + jitter * np.eye(Kmm.shape[0])
    Knm = Kuf(X, kernel, z)
    print('Kmm')
    print(Kmm.shape)
    print('Knm')
    print(Knm.shape)
    chol_mm = sp.linalg.cholesky(Kmm, lower=True)
    print('chol_mm.shape')
    print(chol_mm.shape)
    iK_qmu = sp.linalg.solve_triangular(
        chol_mm.T, sp.linalg.solve_triangular(chol_mm, q_mu, lower=True))
    print("iK_qmu.shape")
    print(iK_qmu.shape)
    mu_h = mean_func + Knm @ iK_qmu
    # mu_h = Knm @ iK_qmu
    print('mu_h')
    # print(mu_h)
    print(mu_h.shape)

    Kmm = Kuu(z, kernel)
    Kmn = Kuf(z, kernel, X)
    Knn = kernel.K(X, X)
    Lm = sp.linalg.cholesky(Kmm, lower=True)
    A = sp.linalg.solve_triangular(Lm, Kmn, lower=True)
    fmean = A.T @ q_mu
    fmean = fmean + mean_func
    mu_h = fmean

    dkxs = jacfwd(Ksparse, 1)(X, x_star, z, q_mu, q_sqrt, kernel, mean_func)
    print('dkxs')
    print(dkxs.shape)
    dkxs = dkxs.reshape(num_data, 2)
    print('dkxs')
    print(dkxs.shape)

    # TODO should both derivatives be wrt to x_star???
    dksx = dkxs.T
    # dksx = jacfwd(Ksparse, 0)(x_star, X, z, q_mu, q_sqrt, kernel, mean_func)
    # print('dksx')
    # print(dksx.shape)
    # dksx = dksx.reshape(num_data, 2)
    # print('dkxs')
    # print(dksx.shape)

    # iKh = sp.linalg.solve_triangular(
    #     chol.T, sp.linalg.solve_triangular(chol, mu_h, lower=True))
    print('aaaa')
    # print(m_h_mu)
    print(m_h_mu.shape)
    print(chol.shape)
    m_h_mu += mean_func
    iKh = sp.linalg.solve_triangular(
        chol.T, sp.linalg.solve_triangular(chol, m_h_mu, lower=True))
    print("iKh.shape")
    print(iKh.shape)
    mu_j = dkxs.T @ iKh
    print('mu_j')
    print(mu_j.shape)

    # # Lm = sp.linalg.cholesky(Kmm, lower=True)
    # A = sp.linalg.solve_triangular(chol, dkxs, lower=True)
    # print('new A')
    # print(A.shape)
    # mu_j = A.T @ mu_h
    # print('new mu_j')
    # print(mu_j.shape)

    v = sp.linalg.solve_triangular(chol, dkxs, lower=True)

    # l2 = kernel.lengthscale**2
    # l2 = np.diag(l2)
    # d2k_dtt = -l2 * kernel.K(xy, xy)

    # calculate mean and variance of J
    # print("kinvy")
    # print(kinvy.shape)
    # mu_j = np.dot(dk_dtT, kinvy)
    # mu_j = mean_func * np.ones([2, 1])
    # print("mu_j")
    # print(mu_j.shape)
    # TODO does all of d2k need to be calculated for sparse GP
    cov_j = d2k - np.matmul(v.T, v)  # d2K doesn't need to be calculated
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

    mu_jT = mu_j.T
    # assert mu_jT.shape == (1, input_dim)
    print('here')
    print(mu_j.shape)
    print(mu_jT.shape)

    jTj = np.matmul(mu_j, mu_jT)  # [input_dim x input_dim]
    assert jTj.shape == (input_dim, input_dim)
    var_weight = 1.
    # var_weight = 0.1
    G = jTj + var_weight * output_dim * cov_j  # [input_dim x input_dim]
    assert G.shape == (input_dim, input_dim)
    return G, mu_j, cov_j


# @partial(jit, static_argnums=(1, 2, 3))
# def single_gp_derivative_predict(xy, X, Y, kernel):
#     Kxx = kernel.K(X, X)
#     Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
#     chol = sp.linalg.cholesky(Kxx, lower=True)
#     # TODO check cholesky is implemented correctly
#     kinvy = sp.linalg.solve_triangular(
#         chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))

#     dk_dt0 = kernel.dK_dX(xy, X, 0)
#     dk_dt1 = kernel.dK_dX(xy, X, 1)
#     dk_dtT = np.stack([dk_dt0, dk_dt1], axis=1)
#     dk_dtT = np.squeeze(dk_dtT)

#     v = sp.linalg.solve_triangular(chol, dk_dtT.T, lower=True)

#     l2 = kernel.lengthscale**2
#     l2 = np.diag(l2)
#     d2k_dtt = -l2 * kernel.K(xy, xy)

#     # calculate mean and variance of J
#     mu_j = np.dot(dk_dtT, kinvy)
#     cov_j = d2k_dtt - np.matmul(v.T, v)  # d2Kd2t doesn't need to be calculated
#     return mu_j, cov_j

# def gp_derivative_predict(xy, X, Y, kernel):
#     """
#     xy: test inputs to calulate predictive mean and variance at [M, num_data]
#     X: training inputs [num_data, input_dim]
#     Y: training outputs [num_data, output_dim]
#     """
#     input_dim = X.shape[1]
#     print('xy.shape')
#     print(xy.shape)
#     print(input_dim)
#     if xy.shape == (961, input_dim):
#         mu_j, cov_j = vmap(single_gp_derivative_predict,
#                            in_axes=(0, None, None, None))(xy, X, Y, kernel)
#     else:
#         raise ValueError(
#             'Test inputs array should be of shape (-1, input_dim)')
#     return mu_j, cov_j

# @partial(jit, static_argnums=(1, 2, 3))
# def calc_G_map(c, X, Y, kernel):
#     c = c.reshape(1, -1)
#     input_dim = X.shape[1]
#     output_dim = Y.shape[1]
#     mu_j, cov_j = single_gp_derivative_predict(c, X, Y, kernel)

#     mu_jT = mu_j.T
#     # assert mu_jT.shape == (1, input_dim)

#     jTj = np.matmul(mu_j, mu_jT)  # [input_dim x input_dim]
#     assert jTj.shape == (input_dim, input_dim)
#     var_weight = 0.1
#     G = jTj + var_weight * output_dim * cov_j  # [input_dim x input_dim]
#     assert G.shape == (input_dim, input_dim)
#     return G, mu_j, cov_j


def Kuu(inducing_inputs, kernel, jitter=1e-4):
    Kzz = kernel.K(inducing_inputs, inducing_inputs)
    Kzz += jitter * np.eye(len(inducing_inputs), dtype=Kzz.dtype)
    return Kzz


def Kuf(inducing_inputs, kernel, Xnew):
    return kernel.K(inducing_inputs, Xnew)


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
    # fvar = Knn + LTA.T @ LTA
    # fvar = fvar + LTA.T @ LTA

    return fmean, fvar


def gp_predict_sparse_LTA(x_star,
                          z,
                          mean_func,
                          q_mu,
                          q_sqrt,
                          kernel,
                          jitter=1e-8):
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
    # fvar = fvar + LTA.T @ LTA

    return fmean, fvar


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


def plot_contour(ax, x, y, z, a=None, contour=None):
    # surf = ax.contourf(x, y, z, cmap=cm.cool, linewidth=0, antialiased=False)
    print('inside')
    print(x.shape)
    print(y.shape)
    print(z.shape)
    surf = ax.contourf(x,
                       y,
                       z,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
    cont = None
    if contour is not None and a is not None:
        cont = ax.contour(x, y, a, levels=contour)
    if contour is not None and a is None:
        cont = ax.contour(x, y, z, levels=contour)
    # if inputs is True:
    #     # plt.plot(x.flatten(), y.flatten(), 'xk')
    #     plt.plot(X_[:, 0], X_[:, 1], 'xk')
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    return surf, cont


def plot_mean_and_var_contour(x, y, z_mu, z_var, a=None, a_true=None,
                              title=""):
    # fig, axs = plt.subplot(2, sharex=True, sharey=True)
    fig, axs = plt.subplots(1, 2, figsize=(24, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    surf_mu, cont_mu = plot_contour(axs[0], x, y, z_mu, a)
    cbar = fig.colorbar(surf_mu, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label('mean')
    surf_var, cont_var = plot_contour(axs[1], x, y, z_var, a)
    cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label('variance')
    plt.suptitle(title)

    # if a_true is not None:
    #     axs[0].plot(a_true[0], a_true[1], 'k')
    #     axs[1].plot(a_true[0], a_true[1], 'k')

    return axs


if __name__ == "__main__":
    # x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 1000
    # y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 1000
    # global m_h_mu

    X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var = load_data_and_init_kernel(
        filename='saved_models/27-2/137/params_from_model.npz')
    # filename='saved_models/26-2/1247/params_from_model.npz')
    # filename='saved_models/20-2/189/params_from_model.npz')

    # def inv_probit(x):
    #     jitter = 1e-3  # ensures output is strictly between 0 and 1
    #     return 0.5 * (1.0 + sp.special.erf(x / np.sqrt(2.0))) \
    #         * (1 - 2 * jitter) + jitter

    # xy, xx, yy = create_grid(X, N=961)
    # xy, xx, yy = create_grid(X, N=100)
    # print('Calculating trace of metric, cov_j and mu_j...')
    # G, mu_j, cov_j = vmap(calc_G_map, in_axes=(0, None, None, None))(xy, X, Y,
    #                                                                  kernel)
    # print('Done calculating metric')

    mu_j_sparse, cov_j_sparse = sparse_gp_derivative_predict(
        xy, X, z, q_mu, q_sqrt, kernel, mean_func, m_h_mu)
    print('sfas')

    # axs = plot_mean_and_var(xy, mu, var)
    print(h_mu.shape)
    print(h_var.shape)
    print(q_mu.shape)
    print(q_sqrt.shape)
    print(X.shape)
    print(Y.shape)
    print(xx.shape)
    print(yy.shape)
    print(xy.shape)
    axs = plot_mean_and_var_contour(xx, yy, h_mu.reshape(xx.shape),
                                    h_var.reshape(xx.shape))
    plt.suptitle('h learned and predicted from BMNSVGP')
    # axs = plot_mean_and_var_contour(xy[:, 0], xy[:, 1], h_mu, h_var)

    mu, var = gp_predict(xy, X, m_h_mu, kernel, mean_func=mean_func)
    var = np.diag(var).reshape(-1, 1)
    axs = plot_mean_and_var(xy, mu, var)
    plt.suptitle("Full GP prediction func")

    # print(z.shape)
    # z[:, [0, 1]] = z[:, [1, 0]]
    mu, var = gp_predict_sparse(xy,
                                z,
                                mean_func,
                                q_mu,
                                q_sqrt,
                                kernel,
                                jitter=1e-4)
    var = np.diag(var).reshape(-1, 1)
    axs_sparse = plot_mean_and_var(xy, mu, var)
    plt.scatter(z[:, 0], z[:, 1])
    plt.suptitle("Sparse GP prediction func")

    mu, var = gp_predict_sparse(xy,
                                z,
                                mean_func,
                                q_mu,
                                q_sqrt,
                                kernel,
                                jitter=1e-4)
    var = np.diag(var).reshape(-1, 1)
    axs = plot_mean_and_var_contour(xx, yy, mu.reshape(xx.shape),
                                    var.reshape(xx.shape))
    plt.suptitle("Sparse GP prediction func")

    mu, var = gp_predict_sparse_LTA(xy,
                                    z,
                                    mean_func,
                                    q_mu,
                                    q_sqrt,
                                    kernel,
                                    jitter=1e-4)
    var = np.diag(var).reshape(-1, 1)
    axs = plot_mean_and_var_contour(xx, yy, mu.reshape(xx.shape),
                                    var.reshape(xx.shape))
    plt.suptitle("Sparse GP prediction func with LTA.T LTA")
    plt.show()
    # plt.savefig(save_dirname + 'alpha.pdf', transparent=True)

    #     # plot original GP
    #     xy, xx, yy = create_grid(X, N=961)
    #     mu, var = gp_predict(xy, X, a_mu, kernel)
    #     var = np.diag(var).reshape(-1, 1)
    #     axs = plot_mean_and_var(xy, mu, var)
    #     plt.suptitle('Original GP')

    #     xy, xx, yy = create_grid(X, N=961)
    # axs = plot_gradeint(xy, X, Y, kernel)
#     axs = plot_metric_trace(xy, X, Y, kernel)

#     plt.show()
