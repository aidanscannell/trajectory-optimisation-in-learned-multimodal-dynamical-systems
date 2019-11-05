# Copyright 2019 Aidan Scannell

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.stats import multivariate_normal
from utils import plot_contourf, plot_data_1d


def func1(X):
    return X[:, 0]**2 + X[:, 1]**2


def func2(X):
    return X[:, 0]**2 - X[:, 1]**2


def func(x):
    return np.sin(
        x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)


def gen_data(N=600,
             input_dim=1,
             output_dim=1,
             low_lim=-1,
             high_lim=1,
             low_noise_var=0.005,
             high_noise_var=0.3,
             func_list=None,
             plot_flag=False):
    """
    This function creates datasets with either 1D/2D inputs and 1D/2D outputs
    depending on the functions passed in func_list.

    Parameters
    ----------
        N - number of training observations (if 2d input N should be a squre number)
        input_dim - dimensionality of inputs
        output_dim - dimensionality of outputs
        low_lim - lowest number in all dimensions of input
        high_lim - highest number in all dimensions of input
        low_noise_var - variance used to add iid Gaussian noise to low noise regions
        high_noise_var - variance used to add iid Gaussian noise to high noise regions
        func_list - list of functions, one for each output dimension.
                    length=output_dimension,
                    each function should take a single input with shape [N x input_dim]
        plot_flag - True to plot dataset

    Returns
    -------
        X - inputs [N x input_dim]
        Y - outputs [N x output_dim]
    """
    if input_dim == 1:
        X = np.sort(np.random.rand(N, input_dim) * (high_lim - low_lim) +
                    low_lim,
                    axis=0)
    elif input_dim == 2:
        # TODO: higher order root depending on input_dim
        sqrtN = int(np.sqrt(N))
        X = np.sort(np.random.rand(sqrtN, input_dim) * (high_lim - low_lim) +
                    low_lim,
                    axis=0)
        x, y = np.meshgrid(X[:, 0], X[:, 1])
        X = np.column_stack([x.flat,
                             y.flat])  # Need an (N, 2) array - N (x, y) pairs.

    # Calculate uncorrupted output using funcs_list
    Y = np.stack([f(X).flatten() for f in func_list], axis=1)

    # Generate alpha (separation) from a mixture of gaussians
    num_mixtures = 2
    means = np.random.rand(num_mixtures, input_dim) * 2 - 1
    covs = np.random.rand(num_mixtures, input_dim, input_dim) * 0.5
    covs = [np.diag(np.diag(covs[i, :, :])) for i in range(num_mixtures)]
    a = sum([
        multivariate_normal.pdf(X, mean=mean, cov=cov)
        for mean, cov in zip(means, covs)
    ]).reshape(N, 1)
    a = np.interp(a, (a.min(), a.max()), (0, +1))  # rescale to range 0, 1

    # Add Gaussian noise where \alpha>0.5
    aa = np.tile(a, Y.shape[1])  # broadcast a along output dimension
    Y = np.where(aa > 0.5, Y + high_noise_var * np.random.randn(*Y.shape),
                 Y + low_noise_var * np.random.randn(*Y.shape))

    if plot_flag is True:
        if input_dim == 1 and output_dim == 1:
            plot_data_1d(X, Y, a, func)
        elif input_dim == 2 and output_dim == 1:
            plot_contourf(x,
                          y,
                          a.reshape(sqrtN, sqrtN),
                          contour=[0.5],
                          title='alpha')
            plot_contourf(x,
                          y,
                          Y.reshape(x.shape),
                          a.reshape(sqrtN, sqrtN),
                          contour=[0.5],
                          title='y - noisy')
        elif input_dim == 2 and output_dim == 2:
            plot_contourf(x,
                          y,
                          a.reshape(sqrtN, sqrtN),
                          contour=[0.5],
                          title='alpha')
            plot_contourf(x,
                          y,
                          Y[:, 0].reshape(x.shape),
                          a.reshape(sqrtN, sqrtN),
                          contour=[0.5],
                          title='$y_1$ - noisy')
            plot_contourf(x,
                          y,
                          Y[:, 1].reshape(x.shape),
                          a.reshape(sqrtN, sqrtN),
                          contour=[0.5],
                          title='$y_2$ - noisy')

    return X, Y, a
