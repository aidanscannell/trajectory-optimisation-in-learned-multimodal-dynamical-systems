from jax import numpy as np
from jax import scipy as sp


def gp_predict(x_star, X, Y, kernel, mean_func=0., jitter=1e-4):
    num_data = X.shape[0]

    Kxx = kernel.K(X, X)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    assert chol.shape == (num_data, num_data)
    kinvy = sp.linalg.solve_triangular(
        chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))
    assert kinvy.shape == (num_data, 1)

    # calculate mean and variance of J
    Kxs = kernel.K(X, x_star)
    mu = np.dot(Kxs.T, kinvy)

    Kss = kernel.K(x_star, x_star)
    v = sp.linalg.solve_triangular(chol, Kxs, lower=True)
    vT = v.T
    cov = Kss - np.matmul(vT, v)
    mu = mu + mean_func
    return mu, cov
