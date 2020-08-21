from jax import numpy as np
from ProbGeo.kernels import DiffRBF


def load_data_and_init_kernel_fake(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    l = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    Y = params['a_mu']  # [num_data x 1] mean of alpha
    # a_var = params['a_var']  # [num_data x 1] variance of alpha
    kernel = DiffRBF(X.shape[1], variance=var, lengthscale=l, ARD=True)
    return X, Y, kernel


def load_data_and_init_kernel_sparse(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscale = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    Z = params['z']  # [num_data x 2]
    q_mu = params['q_mu']  # [num_data x 1] mean of alpha
    q_sqrt = params['q_sqrt']  # [num_data x 1] variance of alpha
    mean_func = params['mean_func']
    kernel = DiffRBF(X.shape[1],
                     variance=var,
                     lengthscale=lengthscale,
                     ARD=True)
    return X, Z, q_mu, q_sqrt, kernel, mean_func
