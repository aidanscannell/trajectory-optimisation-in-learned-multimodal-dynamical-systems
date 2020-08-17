from jax import numpy as np
from jax import scipy as sp
from ProbGeo.derivative_kernel_gpy import DiffRBF


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
