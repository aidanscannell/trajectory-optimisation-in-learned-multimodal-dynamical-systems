import numpy as np
from scipy.stats import multivariate_normal

from utils import plot_contourf

func1 = lambda x, y: x**2 + y**2
func2 = lambda x, y: x**2 - y**2


def gen_data_2d(N=100,
                x_scale=1.,
                y_scale=1.,
                num_mixtures=5,
                low_noise_var=0.005,
                high_noise_var=0.3,
                plot_flag=[0, 0, 0]):
    N = int(np.sqrt(N))
    # x_ = np.random.rand(N, 1) * 2 - 1 # X values
    x_ = np.random.rand(N + 10, 1) * 2 * x_scale - 1 * x_scale  # X values
    y_ = np.random.rand(N + 10, 1) * 2 * y_scale - 1 * y_scale  # X values

    x_ = np.sort(x_, axis=0)
    y_ = np.sort(y_, axis=0)

    x_ = np.delete(x_, np.arange(5, 15), 0)
    y_ = np.delete(y_, np.arange(0, 10), 0)

    x, y = np.meshgrid(x_, y_)
    # x, y = np.mgrid[x_min:x_max:N*1j, x_min:x_max:N*1j]
    xy = np.column_stack([x.flat,
                          y.flat])  # Need an (N, 2) array - N (x, y) pairs.

    # generate alpha (separation) from a mixture of gaussians
    means = np.random.rand(num_mixtures, 2) * 2 - 1
    covs = np.random.rand(num_mixtures, 2, 2) * 0.5
    covs = [np.diag(np.diag(covs[i, :, :])) for i in range(num_mixtures)]
    a = sum([
        multivariate_normal.pdf(xy, mean=mean, cov=cov)
        for mean, cov in zip(means, covs)
    ])

    # Reshape back to a (N, N) grid and rescale to range 0, 1
    a = a.reshape(x.shape)
    a = np.interp(a, (a.min(), a.max()), (0, +1))

    # Generate outputs using f(x, y) = x**2 + y**2
    z = np.zeros([N**2, 2])
    z[:, 0:1] = func1(x.reshape(-1, 1), y.reshape(-1, 1))
    z[:, 1:2] = func2(x.reshape(-1, 1), y.reshape(-1, 1))
    # if plot_flag[1] is 1:
    #     plot_contourf(x, y, z[:, 0].reshape(x.shape), title='y1 - no noise')
    #     plot_contourf(x, y, z[:, 1].reshape(x.shape), title='y2 - no noise')

    # a = np.zeros(x.shape)
    # a[:10, :10] = 1
    if plot_flag[0] is 1:
        plot_contourf(x, y, a, contour=[0.5], title='alpha')

    # Add noise where alpha > 0.5
    aa = np.tile(a.reshape(-1, 1),
                 z.shape[1])  # broadcast a along output dimension
    z = np.where(aa > 0.5, z + high_noise_var * np.random.randn(*z.shape),
                 z + low_noise_var * np.random.randn(*z.shape))
    if plot_flag[2] is 1:
        plot_contourf(x,
                      y,
                      z[:, 0].reshape(x.shape),
                      a,
                      contour=[0.5],
                      title='y - noisy')
        plot_contourf(x,
                      y,
                      z[:, 1].reshape(x.shape),
                      a,
                      contour=[0.5],
                      title='y - noisy')

    return xy, z.reshape(-1, 2), a.reshape(-1, 1)
