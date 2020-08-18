import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
from matplotlib import cm
from scipy.stats import multivariate_normal

import gpflow
from utils.gen_data import func, func1, func2, gen_data
from gpflow.params import ParamList
from utils.utils import plot_contourf, plot_loss, plot_model_2d_uav, run_adam


def plot_contour(ax, xx, yy, z):
    surf = ax.contourf(xx,
                       yy,
                       z.reshape(xx.shape),
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
    return surf


# generate data set
input_dim = 2
N = 961  # number of training observations

low_lim = -3
high_lim = 3
# TODO: higher order root depending on input_dim
sqrtN = int(np.sqrt(N))
X = np.sort(np.random.rand(sqrtN, input_dim) * (high_lim - low_lim) + low_lim,
            axis=0)
x, y = np.meshgrid(X[:, 0], X[:, 1])
X = np.column_stack([x.flat, y.flat])  # Need an (N, 2) array - N (x, y) pairs.

# Generate alpha (separation) from a mixture of gaussians
num_mixtures = 2
means = np.array([[1.5, -1.], [1.5, -1.]])
covs = np.random.rand(num_mixtures, input_dim, input_dim) * 0.5
covs = np.array([[[1., 0.], [0., 1.]], [[0.1, 0.], [0., 0.1]]])
covs = [np.diag(np.diag(covs[i, :, :])) for i in range(num_mixtures)]
a = sum([
    multivariate_normal.pdf(X, mean=mean, cov=cov)
    for mean, cov in zip(means, covs)
]).reshape(N, 1)
a = np.interp(a, (a.min(), a.max()), (0, +1))  # rescale to range 0, 1

plot_contourf(x, y, a.reshape(sqrtN, sqrtN), contour=[0.5], title='alpha')

# remove some data points
x1_low = -1.5
x2_low = -0.5
x1_high = 0.
x2_high = 2.
mask_0 = X[:, 0] < x1_low
mask_1 = X[:, 1] < x2_low
mask_2 = X[:, 0] > x1_high
mask_3 = X[:, 1] > x2_high
mask = mask_0 | mask_1 | mask_2 | mask_3
X_partial = X[mask, :]
Y_partial = a[mask, :]

plt.scatter(X_partial[:, 0], X_partial[:, 1], color='k', marker='x')
plt.show()

num_data = X_partial.shape[0]
input_dim = X_partial.shape[1]

lik = gpflow.likelihoods.Gaussian(0)
kern = gpflow.kernels.SquaredExponential(input_dim, ARD=True)
kern.lengthscales = [0.4, 0.4]
# m = gpflow.models.GPR(X_partial, Y_partial, kern=kern)
M = 200
idx = np.random.choice(range(num_data), size=M, replace=False)
feat = None
Z = X[idx, ...].reshape(-1, input_dim)
feature = gpflow.features.inducingpoint_wrapper(feat, Z)
feature.trainable = False
mean_func = gpflow.mean_functions.Constant()
m = gpflow.models.SVGP(X_partial,
                       Y_partial,
                       kern=kern,
                       mean_function=mean_func,
                       likelihood=lik,
                       feat=feature)
m.likelihood.variance = 0.
m.likelihood.variance.trainable = False
# l = [0.01, 0.01]
# l = ParamList(l)
# m.kern.lengthscales = l
m.compile()

num_iters = 2000
logger = run_adam(m, maxiter=gpflow.test_util.notebook_niter(num_iters))
print(m.as_pandas_table())
plot_loss(logger)

# x1_high = X[:, 0].max()
# x2_high = X[:, 1].max()
# x1_low = X[:, 0].min()
# x2_low = X[:, 1].min()
x1_high = 4
x2_high = 4
x1_low = -4
x2_low = -4
x1_high = 3
x2_high = 3
x1_low = -3
x2_low = -3
N = np.sqrt(N)
xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
xy = np.column_stack([xx.flat, yy.flat])

Y_mu, Y_var = m.predict_y(xy)

fig, axs = plt.subplots(1, 2, figsize=(24, 4))
plt.subplots_adjust(wspace=0, hspace=0)
surf_mean = plot_contour(axs[0], xx, yy, Y_mu)
surf_var = plot_contour(axs[1], xx, yy, Y_var)
cbar = fig.colorbar(surf_mean, shrink=0.5, aspect=5, ax=axs[0])
cbar.set_label('mean')
cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
cbar.set_label('variance')

plt.scatter(X_partial[:, 0], X_partial[:, 1], color='k', marker='x')
# for ax in axs:
#     ax.scatter(c[0, 0], c[0, 1], marker='x', color='k')
#     ax.scatter(start[0, 0], start[0, 1], color='k', marker='x')
#     ax.annotate("start", (start[0, 0], start[0, 1]))
#     ax.scatter(end[0], end[1], color='k', marker='x')
#     ax.annotate("end", (end[0], end[1]))

plt.show()

lengthscales = m.kern.lengthscales.value
variance = m.kern.variance.value
X = m.X.value
alpha = m.Y.value

q_mu = m.q_mu.value
q_sqrt = m.q_sqrt.value
z = m.feature.Z.value
mean_func = m.mean_function.c.value
print(X.shape)
print(alpha.shape)
print(lengthscales)
print(variance)

np.savez('./saved_models/params_fake_sparse',
         l=lengthscales,
         var=variance,
         x=X,
         mean_func=mean_func,
         q_mu=q_mu,
         q_sqrt=q_sqrt,
         z=z)
