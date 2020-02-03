# import scipy
import jax.scipy as scipy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from derivative_kernel_gpy import DiffRBF
# from geodesic_func import calc_G_map, load_data_and_init_kernel
from geodesic_func import calc_G_map


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


def plot_mean_and_var(X, Y_mean, Y_var, N=961):
    fig, axs = plt.subplots(1, 2, figsize=(24, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    surf_mean = plot_contour(X, Y_mean, N, ax=axs[0])
    surf_var = plot_contour(X, Y_var, N, ax=axs[1])
    cbar = fig.colorbar(surf_mean, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label('mean')
    cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label('variance')
    return axs


def plot_contour(X, Y, N, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    x1_high = X[:, 0].max()
    x2_high = X[:, 1].max()
    x1_low = X[:, 0].min()
    x2_low = X[:, 1].min()
    N = np.sqrt(N)
    xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
    xy = np.column_stack([xx.flat, yy.flat])

    surf = ax.contourf(xx,
                       yy,
                       Y.reshape(xx.shape),
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
    return surf


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


# import data and parameters
filename = 'saved_models/params.npz'
X, Y, kernel = load_data_and_init_kernel(filename)
params = np.load(filename)
y = params['a']  # [num_data x 2] meen and var of alpha
# Y = y[0:1, :, 0].T  # [num_data x 1]

# plot mean and variance of learned alpha
Y_mean = y[0:1, :, 0].T
Y_var = y[1:2, :, 0].T
N = 961  # number of training observations
# plot_mean_and_var(X, Y_mean, Y_var, N)
# plt.show()

# predict mean and variance of GP at a set of grid locations
x1_high = X[:, 0].max()
x2_high = X[:, 1].max()
x1_low = X[:, 0].min()
x2_low = X[:, 1].min()
sqrtN = np.sqrt(N)
xx, yy = np.mgrid[x1_low:x1_high:sqrtN * 1j, x2_low:x2_high:sqrtN * 1j]
xy = np.column_stack([xx.flat, yy.flat])
mu, var = gp_predict(xy, X, Y, kernel)
var = np.diag(var)
# plot_mean_and_var(X, mu, var, N=N)
# plt.show()

# remove some data points
# mask_0 = X[:, 0] < 0.
# mask_1 = X[:, 1] > 1.4
# mask = mask_0 | mask_1
mask_0 = X[:, 0] < -0.5
mask_1 = X[:, 1] < -0.4
mask_2 = X[:, 0] > 1.5
mask_3 = X[:, 1] > 1.2
mask = mask_0 | mask_1 | mask_2 | mask_3
# mask = mask_0 & mask_1 & mask_2 & mask_3
# mask = mask_0
X_partial = X[mask, :]
Y_partial = Y[mask, :]
Y_partial = np.zeros(Y_partial.shape)
# X_partial = X
# Y_partial = Y
# Y_partial = np.zeros(Y.shape)
mu, var = gp_predict(xy, X_partial, Y_partial, kernel)
var = np.diag(var)
axs = plot_mean_and_var(X, mu, var)
plt.scatter(X_partial[:, 0], X_partial[:, 1], color='k', marker='x')
# plt.show()

# plot_metric = True
plot_metric = False
# calc_metric_at_point = True
calc_metric_at_point = False

calc_length = True
# calc_length = False
plot_metric_trace = True
# plot_metric_trace = False

if plot_metric_trace:
    traceG = np.zeros([xy.shape[0], 1])
    for i in range(xy.shape[0]):
        G, _, _ = calc_G_map(xy[i, :].reshape(1, -1), X_partial, Y_partial,
                             kernel)
        traceG[i] = np.trace(G)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    surf_x = ax.contourf(xx,
                         yy,
                         traceG.reshape(xx.shape),
                         cmap=cm.coolwarm,
                         antialiased=False)
    plt.title('Trace of G(x)')
    # plt.show()

# TODO a and b should be in tangent space
if plot_metric:
    aGbs = []
    aGb = np.zeros([xy.shape[0], 2])
    for i, (x1, x2) in enumerate(xy):
        x = np.array([x1, x2]).reshape(1, -1)
        start = [
            x + np.array([0.2, 0]).reshape(1, -1),
            x + np.array([0., 0.2]).reshape(1, -1)
        ]
        end = [
            x.T + np.array([-0.2, 0]).reshape(-1, 1),
            x.T + np.array([0., -0.2]).reshape(-1, 1)
        ]

        # G, mu_j, var_j = calc_G_map(x, X_partial, Y_partial, kernel)
        G, mu_j, var_j = calc_G_map(x)
        for j, (a, b) in enumerate(zip(start, end)):
            aGb[i, j] = a @ G @ b

    Yx = aGb[:, 0]
    Yy = aGb[:, 1]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    surf_x = axs[0].contourf(xx,
                             yy,
                             Yx.reshape(xx.shape),
                             cmap=cm.coolwarm,
                             antialiased=False)
    surf_y = axs[1].contourf(xx,
                             yy,
                             Yy.reshape(xx.shape),
                             cmap=cm.coolwarm,
                             antialiased=False)

    cbar = fig.colorbar(surf_x, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label('$a^TG(x)b$ - x')
    cbar = fig.colorbar(surf_y, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label('$a^TG(x)b$ - y')
    plt.show()

if calc_length:
    # x = np.array([0., -0.4]).reshape(1, -1)
    x = np.array([0.6, 0.5]).reshape(1, -1)
    start = np.array([[-0.5, 0.5], [-.5, 1.2], [-.5, -.5]])
    end = np.array([[1.5, 0.5], [1.5, 1.2], [1.5, -.5]])
    end = np.array([[1.5, 0.6], [1.5, 1.3], [1.5, -.4]])

    # G, mu_j, var_j = calc_G_map(x.reshape(1, -1), X_partial, Y_partial, kernel)

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
            # G, _, _ = calc_G_map(cs[i, :].reshape(1, -1), X_partial, Y_partial,
            #                      kernel)
            # print('cs[i,:]')
            # print(cs[i, :])
            G, mu_j, var_j = calc_G_map(cs[i, :].reshape(1, -1), X_partial,
                                        Y_partial, kernel)
            dcdt = (cs[i, :] - cs[i - 1, :]) / dt
            dcdt = dcdt.reshape(-1, 1)
            dcMdc = dcdt.T @ G @ dcdt
            length += dcMdc
            print(G)
            print(length)
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

# if calc_metric_at_point:
#     x = np.array([0., -0.4]).reshape(1, -1)
#     start = [
#         x + np.array([0.2, 0]).reshape(1, -1),
#         x + np.array([0., 0.2]).reshape(1, -1)
#     ]
#     end = [
#         x.T + np.array([-0.2, 0]).reshape(-1, 1),
#         x.T + np.array([0., -0.2]).reshape(-1, 1)
#     ]
#     # end = c + np.array([-0.2, 0]).reshape(1, -1)
#     # end = end.T

#     G, mu_j, var_j = calc_G_map(x, X_partial, Y_partial, kernel)
#     # print(c)
#     # print(G)
#     print('mu_j')
#     print(mu_j)
#     print('mu_j^T @ mu_j')
#     print(mu_j @ mu_j.T)
#     print('var_j')
#     print(var_j)
#     print("G")
#     print(G)

#     axs = plot_mean_and_var(X, mu, var)
#     for a, b in zip(start, end):
#         aGb = a @ G @ b
#         print(
#             'aTG(x)b = %.5f for a:(%.2f, %.2f), b:(%.2f, %.2f), x:(%.2f, %.2f)'
#             % (aGb, a[0, 0], a[0, 1], b[0], b[1], x[0, 0], x[0, 1]))

#         # axs = plot_mean_and_var(X_partial, mu, var)
#         plt.scatter(X_partial[:, 0], X_partial[:, 1], color='k', marker='x')
#         for ax in axs:
#             ax.scatter(x[0, 0], x[0, 1], marker='x', color='k')
#             ax.scatter(a[0, 0], a[0, 1], color='k', marker='x')
#             ax.annotate("start", (a[0, 0], a[0, 1]))
#             ax.scatter(b[0], b[1], color='k', marker='x')
#             ax.annotate("end", (b[0], b[1]))

#     plt.show()
