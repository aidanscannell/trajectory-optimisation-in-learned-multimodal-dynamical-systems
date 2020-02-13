# import scipy
# import numpy as np
import jax.numpy as np
import jax.scipy as scipy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from jax import vmap
from matplotlib import cm

from derivative_kernel_gpy import DiffRBF
# from geodesic_func import calc_G_map, load_data_and_init_kernel
# from geodesic_func import calc_G_map
from geodesic_func_jac_and_val import calc_G_map


def load_data_and_init_kernel(
        filename='saved_models/6-2/params_from_model.npz'):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscale = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    a_mu = params['a_mu']  # [num_data x 1] mean of alpha
    a_var = params['a_var']  # [num_data x 1] variance of alpha
    kernel = DiffRBF(X.shape[1],
                     variance=var,
                     lengthscale=lengthscale,
                     ARD=True)
    return X, a_mu, a_var, kernel


def plot_mean_and_var(X, Y_mean, Y_var):
    fig, axs = plt.subplots(1, 2, figsize=(24, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    surf_mean = plot_contour(X, Y_mean, ax=axs[0])
    surf_var = plot_contour(X, Y_var, ax=axs[1])
    cbar = fig.colorbar(surf_mean, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label('mean')
    cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label('variance')
    return axs


def plot_contour(X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    cont = ax.tricontourf(X[:, 0], X[:, 1], Y.reshape(-1), 15)
    return cont


def gp_predict(x_star, X, Y, kernel, jitter=1e-4):
    num_data = X.shape[0]
    # input_dim = X.shape[1]

    # standardise input
    # X = (X - X.mean()) / X.std()
    # Y = (Y - Y.mean()) / Y.std()

    Kxx = kernel.K(X, X)
    # print(Kxx.shape)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = scipy.linalg.cholesky(Kxx, lower=True)
    assert chol.shape == (num_data, num_data)
    kinvy = scipy.linalg.solve_triangular(
        chol.T, scipy.linalg.solve_triangular(chol, Y, lower=True))
    assert kinvy.shape == (num_data, 1)

    # calculate mean and variance of J
    Kxs = kernel.K(X, x_star)
    # print(x_star.shape)
    # print(Kxs.shape)
    mu = np.dot(Kxs.T, kinvy)
    # print(mu_j.shape)
    # assert mu_j.shape == (input_dim, 1)

    Kss = kernel.K(x_star, x_star)
    v = scipy.linalg.solve_triangular(chol, Kxs, lower=True)
    # assert v.shape == (num_data, input_dim)
    vT = v.T
    cov = Kss - np.matmul(vT, v)
    # assert cov_j.shape == (input_dim, input_dim)
    # cov *= 100
    return mu, cov


def create_grid(X, N):
    # predict mean and variance of GP at a set of grid locations
    x1_high = X[:, 0].max()
    x2_high = X[:, 1].max()
    x1_low = X[:, 0].min()
    x2_low = X[:, 1].min()

    x1_low = -2.
    x1_high = 3.
    x2_low = -3.
    x2_high = 2.
    sqrtN = int(np.sqrt(N))
    xx = np.linspace(x1_low, x1_high, sqrtN)
    yy = np.linspace(x2_low, x2_high, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    return xy, xx, yy


# def gen_data_geodesic_jax(filename='saved_models/6-2/params_from_model.npz'):
#     # import data and parameters
#     # filename = 'saved_models/params.npz'
#     filename = 'saved_models/6-2/params_from_model.npz'
#     X, Y, kernel = load_data_and_init_kernel(filename)
#     params = np.load(filename)
#     # X = params['x']
#     mX = params['mx']
#     a_mu = params['a_mu']
#     a_var = params['a_var']
#     Y = a_mu
#     print(Y.shape)
#     print(X.shape)

#     # plot mean and variance of learned alpha
#     Y_mean = a_mu
#     Y_var = a_var
#     # N = 961  # number of training observations
#     N = a_mu.shape[0]

#     plot_mean_and_var(X, a_mu, a_var, N)
#     plt.show()

#     return X, Y, kernel
# # predict mean and variance of GP at a set of grid locations
# x1_high = X[:, 0].max()
# x2_high = X[:, 1].max()
# x1_low = X[:, 0].min()
# x2_low = X[:, 1].min()
# sqrtN = int(np.sqrt(N))
# xx = np.linspace(x1_low, x1_high, sqrtN)
# yy = np.linspace(x2_low, x2_high, sqrtN)
# xx, yy = np.meshgrid(xx, yy)
# xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
# mu, var = gp_predict(xy, X, Y, kernel)
# var = np.diag(var)
# plot_mean_and_var(xy, mu, var, N=N)
# plt.show()

# remove some data points
# mask_0 = X[:, 0] < -0.5
# mask_1 = X[:, 1] < -0.4
# mask_2 = X[:, 0] > 1.5
# mask_3 = X[:, 1] > 1.2
# mask = mask_0 | mask_1 | mask_2 | mask_3
# mask = mask_0 & mask_1 & mask_2 & mask_3
# mask = mask_0
# X_partial = X[mask, :]
# Y_partial = Y[mask, :]
# Y_partial = np.zeros(Y_partial.shape)
# X_partial = X
# Y_partial = Y
# Y_partial = np.zeros(Y.shape)
# return X_partial, Y_partial, xx, yy, xy, kernel
# return X, Y, xx, yy, xy, kernel

# def gen_data_geodesic():
#     # import data and parameters
#     filename = 'saved_models/params.npz'
#     X, Y, kernel = load_data_and_init_kernel(filename)
#     params = np.load(filename)
#     y = params['a']  # [num_data x 2] meen and var of alpha
#     # Y = y[0:1, :, 0].T  # [num_data x 1]

#     # plot mean and variance of learned alpha
#     Y_mean = y[0:1, :, 0].T
#     Y_var = y[1:2, :, 0].T
#     N = 961  # number of training observations
#     # plot_mean_and_var(X, Y_mean, Y_var, N)
#     # plt.show()

#     # predict mean and variance of GP at a set of grid locations
#     x1_high = X[:, 0].max()
#     x2_high = X[:, 1].max()
#     x1_low = X[:, 0].min()
#     x2_low = X[:, 1].min()
#     sqrtN = np.sqrt(N)
#     xx, yy = np.mgrid[x1_low:x1_high:sqrtN * 1j, x2_low:x2_high:sqrtN * 1j]
#     xy = np.column_stack([xx.flat, yy.flat])
#     mu, var = gp_predict(xy, X, Y, kernel)
#     var = np.diag(var)
#     # plot_mean_and_var(X, mu, var, N=N)
#     # plt.show()

#     # remove some data points
#     # mask_0 = X[:, 0] < 0.
#     # mask_1 = X[:, 1] > 1.4
#     # mask = mask_0 | mask_1
#     mask_0 = X[:, 0] < -0.5
#     mask_1 = X[:, 1] < -0.4
#     mask_2 = X[:, 0] > 1.5
#     mask_3 = X[:, 1] > 1.2
#     mask = mask_0 | mask_1 | mask_2 | mask_3
#     # mask = mask_0 & mask_1 & mask_2 & mask_3
#     # mask = mask_0
#     X_partial = X[mask, :]
#     Y_partial = Y[mask, :]
#     Y_partial = np.zeros(Y_partial.shape)
#     # X_partial = X
#     # Y_partial = Y
#     # Y_partial = np.zeros(Y.shape)
#     return X_partial, Y_partial, xx, yy, xy, kernel

if __name__ == "__main__":
    X, a_mu, a_var, kernel = load_data_and_init_kernel(
        filename='saved_models/6-2/params_from_model.npz')

    X_partial = X
    Y_partial = a_mu
    plot_mean_and_var(X, a_mu, a_var)
    plt.scatter(X_partial[:, 0], X_partial[:, 1], color='k', marker='x')
    # plt.show()
    xy, xx, yy = create_grid(X, N=961)
    # mu, var = gp_predict(xy, X_partial, Y_partial, kernel)
    # var = np.diag(var)
    # axs = plot_mean_and_var(X_partial, mu, var)
    # plt.scatter(X_partial[:, 0], X_partial[:, 1], color='k', marker='x')
    # plt.show()
    print(X_partial.shape)
    print(Y_partial.shape)
    print(xx.shape)
    print(yy.shape)
    print(xy.shape)

    # plot_metric = True
    plot_metric = False
    # calc_metric_at_point = True
    calc_metric_at_point = False

    calc_length = True
    # calc_length = False
    plot_metric_trace = True
    # plot_metric_trace = False

    if plot_metric_trace:
        print('aaa')
        map_G = vmap(calc_G_map, in_axes=(0, None, None, None))
        print('bbb')
        G, mu_j, cov_j = map_G(xy, X_partial, Y_partial, kernel)
        print('ccc')
        print(G.shape)
        print(mu_j.shape)
        print(cov_j.shape)
        traceG = np.trace(G, axis1=1, axis2=2)
        # trace_mu = np.trace(mu_j, axis1=1, axis2=2)
        trace_cov = np.trace(cov_j, axis1=1, axis2=2)
        print('ddd')
        print(traceG.shape)
        # print(trace_mu.shape)
        print(trace_cov.shape)
        # for i in range(xy.shape[0]):
        #     G, _, _ = calc_G_map(xy[i, :].1, -1), X_partial, Y_partial,
        #                          kernel)
        #     traceG[i] = np.trace(G)

        # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        # surf_x = ax.contourf(xx,
        #                      yy,
        #                      traceG.reshape(xx.shape),
        #                      cmap=cm.coolwarm,
        #                      antialiased=False)
        # cbar = fig.colorbar(surf_x, shrink=0.5, aspect=5, ax=ax)
        # plt.title('Trace of G(x)')
        # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        # surf_x = ax.contourf(xx,
        #                      yy,
        #                      trace_cov.reshape(xx.shape),
        #                      cmap=cm.coolwarm,
        #                      antialiased=False)
        # cbar = fig.colorbar(surf_x, shrink=0.5, aspect=5, ax=ax)
        # plt.title('Trace of cov_j(x)')
        # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        # surf_x = ax.contourf(xx,
        #                      yy,
        #                      mu_j[:, 0, 0].reshape(xx.shape),
        #                      cmap=cm.coolwarm,
        #                      antialiased=False)
        # cbar = fig.colorbar(surf_x, shrink=0.5, aspect=5, ax=ax)
        # plt.title('mu_j1(x)')
        # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        # surf_x = ax.contourf(xx,
        #                      yy,
        #                      mu_j[:, 1, 0].reshape(xx.shape),
        #                      cmap=cm.coolwarm,
        #                      antialiased=False)
        # cbar = fig.colorbar(surf_x, shrink=0.5, aspect=5, ax=ax)
        # plt.title('mu_j2(x)')
        mu_sum = np.sum(mu_j, axis=1)
        titles = [
            'Tr(G(x))', 'Tr(cov_j(x))', 'mu_j_1(x)', 'mu_j_2(x)', 'sum_mu_j(x)'
        ]
        xmin = -2.
        xmax = 3.
        ymin = -3.
        ymax = 2.
        for zz, t in zip(
            [traceG, trace_cov, mu_j[:, 0, 0], mu_j[:, 1, 0], mu_sum], titles):
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            surf_x = ax.contourf(xx,
                                 yy,
                                 zz.reshape(xx.shape),
                                 cmap=cm.coolwarm,
                                 antialiased=False)

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            cbar = fig.colorbar(surf_x, shrink=0.5, aspect=5, ax=ax)

            plt.title(t)
        plt.show()

    if calc_length:
        # x = np.array([0., -0.4]).reshape(1, -1)
        x = np.array([0.6, 0.5]).reshape(1, -1)
        start = np.array([[-0.5, 0.5], [-.5, 1.2], [-.5, -.5]])
        end = np.array([[1.5, 0.5], [1.5, 1.2], [1.5, -.5]])
        end = np.array([[1.5, 0.6], [1.5, 1.3], [1.5, -.4]])

        lengths = []
        for j in range(start.shape[0]):
            s = start[j]
            e = end[j]
            print('start: ', s)
            print('end: ', e)
            cs = np.linspace(s, e, 1000)
            dt = cs[1, 0] - cs[0, 0]
            print('dt: ', dt)
            length = 0
            for i in range(1, cs.shape[0]):
                G, mu_j, var_j = calc_G_map(cs[i, :].reshape(1, -1), X_partial,
                                            Y_partial, kernel)
                dcdt = (cs[i, :] - cs[i - 1, :]) / dt
                dcdt = dcdt.reshape(-1, 1)
                dcMdc = dcdt.T @ G @ dcdt
                length += dcMdc
                # print(G)
                # print(length)
                # length += np.sqrt(-dcMdc)
                # length += np.sqrt(dcMdc)
                # print('mu_j', mu_j)
                # print('var_j', var_j)
            lengths.append(length)

        print(lengths)
        axs = plot_mean_and_var(X_partial, mu, var)
        # print(x)
        # print(x.shape)
        # for ax in axs:
        #     ax.scatter(x[0, 0], x[0, 1], color='k', marker='x')
        #     ax.annotate("G(x)", (x[0, 0], x[0, 1]))
        for j in range(start.shape[0]):
            s = start[j]
            e = end[j]
            for ax in axs:
                ax.plot([s[0], e[0]], [s[1], e[1]])
                ax.scatter(s[0], s[1], marker='x', color='k')
                ax.scatter(e[0], e[1], color='k', marker='x')
                ax.annotate("start", (s[0], s[1]))
                ax.annotate("end", (e[0], e[1]))
                ax.annotate(lengths[j], (s[0] + 0.4, s[1]))

    plt.show()
jkj
