from jax import numpy as np
from jax import scipy as sp


def Kuu(inducing_inputs, kernel, jitter=1e-4):
    Kzz = kernel.K(inducing_inputs, inducing_inputs)
    Kzz += jitter * np.eye(len(inducing_inputs), dtype=Kzz.dtype)
    return Kzz


def Kuf(inducing_inputs, kernel, Xnew):
    return kernel.K(inducing_inputs, Xnew)


def gp_predict_sparse_sym(x1,
                          x2,
                          z,
                          mean_func,
                          q_mu,
                          q_sqrt,
                          kernel,
                          jitter=1e-8):
    k_uu = Kuu(z, kernel)
    k_1u = Kuf(x1, kernel, z)
    k_u2 = Kuf(z, kernel, x2)
    k_ff = kernel.K(x1, x2)

    Lu = sp.linalg.cholesky(k_uu, lower=True)
    A1 = sp.linalg.solve_triangular(Lu, k_1u.T, lower=True)
    A2 = sp.linalg.solve_triangular(Lu, k_u2, lower=True)
    ATA = A1.T @ A2

    Ls = np.squeeze(q_sqrt)

    LTA1 = Ls @ A1
    LTA2 = Ls @ A2

    fvar = k_ff - ATA + LTA1.T @ LTA2

    # print('here1234')
    # print(A1.shape)
    # Amu = sp.linalg.solve_triangular(Lu, q_mu, lower=True)
    # print(Amu.shape)
    # fmean = A1.T @ Amu
    fmean = A1.T @ (q_mu - 0.)
    fmean += mean_func

    return fmean, fvar


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
    fvar = fvar + LTA.T @ LTA

    return fmean, fvar


def Kvar_sparse(x1, x2, z, q_sqrt, kernel):
    k_uu = Kuu(z, kernel)  # 200 200
    k_1u = Kuf(x1, kernel, z)  # 961 200
    k_u2 = Kuf(z, kernel, x2)  # 200 961
    k_ff = kernel.K(x1, x2)  # 961 961

    Lu = sp.linalg.cholesky(k_uu, lower=True)
    A1 = sp.linalg.solve_triangular(Lu, k_1u.T, lower=True)
    A2 = sp.linalg.solve_triangular(Lu, k_u2, lower=True)
    ATA = A1.T @ A2

    Ls = np.squeeze(q_sqrt)
    LTA1 = Ls @ A1
    LTA2 = Ls @ A2

    cov = k_ff - ATA + LTA1.T @ LTA2
    # cov = k_ff + LTA1.T @ LTA2
    return cov


def Kmean_sparse(x1, x2, z, q_mu, kernel, mean_func):
    k_uu = Kuu(z, kernel)
    k_1u = Kuf(x1, kernel, z)

    Lu = sp.linalg.cholesky(k_uu, lower=True)
    A1 = sp.linalg.solve_triangular(Lu, k_1u.T, lower=True)

    fmean = A1.T @ (q_mu - 0.)
    fmean += mean_func
    return fmean
