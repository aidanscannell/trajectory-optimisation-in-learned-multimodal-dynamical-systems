from jax import numpy as np
from jax import jacfwd, jacrev


def Kuu(inducing_inputs, kernel, jitter=1e-4):
    Kzz = kernel.K(inducing_inputs, inducing_inputs)
    Kzz += jitter * np.eye(len(inducing_inputs), dtype=Kzz.dtype)
    return Kzz


def Kuf(inducing_inputs, kernel, Xnew):
    return kernel.K(inducing_inputs, Xnew)


def jacobian_cov_fn_wrt_x1(cov_fn, x1, x2):
    """Calculate derivative of cov_fn wrt to x1

    :param cov_fn: covariance function with signature cov_fn(x1, x2)
    :param x1: [1, input_dim]
    :param x2: [num_x2, input_dim]
    """
    dk = jacfwd(cov_fn, (0))(x1, x2)
    # TODO replace squeeze with correct dimensions
    dk = np.squeeze(dk)
    return dk


def hessian_cov_fn_wrt_x1x1(cov_fn, x1):
    """Calculate derivative of cov_fn(x1, x1) wrt to x1

    :param cov_fn: covariance function with signature cov_fn(x1, x1)
    :param x1: [1, input_dim]
    """
    def cov_fn_(x1):
        return cov_fn(x1, x1)

    d2k = jacrev(jacfwd(cov_fn_))(x1)
    # TODO replace squeeze with correct dimensions
    d2k = np.squeeze(d2k)
    return d2k
