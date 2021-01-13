from jax import numpy as np
from jax import scipy as sp
from jax import partial, jit, jacfwd, jacrev, vmap
from typing import Tuple, List

from ProbGeo.covariances import Kuu, Kuf, jacobian_cov_fn_wrt_x1, hessian_cov_fn_wrt_x1x1, hessian_cov_fn_wrt_x1x1_hard_coded
from ProbGeo.conditionals import base_conditional
from ProbGeo.typing import InputData, OutputData, MeanFunc, MeanAndVariance


def gp_predict(Xnew: InputData,
               X: InputData,
               kernels,
               mean_funcs: List[MeanFunc],
               f: OutputData,
               *,
               full_cov: bool = False,
               q_sqrt=None,
               jitter=1e-6,
               white: bool = True) -> MeanAndVariance:
    fmeans, fvars = [], []
    for output_dim, (kernel, mean_func) in enumerate(zip(kernels, mean_funcs)):
        fmu, fvar = gp_predict_single_output(
            Xnew,
            X,
            kernel,
            mean_func,
            f[:, output_dim:output_dim + 1],
            full_cov=full_cov,
            q_sqrt=q_sqrt[output_dim:output_dim + 1, :, :],
            jitter=jitter,
            white=white)
        fmeans.append(fmu)
        fvars.append(fvar)
    fmeans = np.stack(fmeans)
    fvars = np.stack(fvars)
    return fmeans, fvars


def gp_predict_single_output(Xnew: InputData,
                             X: InputData,
                             kernel,
                             mean_func: MeanFunc,
                             f: OutputData,
                             *,
                             full_cov: bool = False,
                             q_sqrt=None,
                             jitter=1e-6,
                             white: bool = True) -> MeanAndVariance:
    # TODO add noise???
    Kmm = Kuu(X, kernel)

    # Kmm += jitter * np.eye(Kmm.shape[0])
    Kmn = kernel.K(X, Xnew)
    if full_cov:
        Knn = kernel.K(Xnew, Xnew)
    else:
        Knn = kernel.Kdiag(Xnew)

    print('q_sqrt')
    print(q_sqrt.shape)
    if q_sqrt is not None:
        # TODO map over output dimension
        q_sqrt = np.squeeze(q_sqrt)
        q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

    # TODO map over output dimension of Y??
    # f += mean_func
    fmean, fvar = base_conditional(Kmn=Kmn,
                                   Kmm=Kmm,
                                   Knn=Knn,
                                   f=f,
                                   full_cov=full_cov,
                                   q_sqrt=q_sqrt,
                                   white=white)
    # return fmean, fvar
    return fmean + mean_func, fvar


# def gp_predict(Xnew: InputData,
#                X: InputData,
#                kernel,
#                mean_func: MeanFunc,
#                f: OutputData,
#                *,
#                full_cov: bool = False,
#                q_sqrt=None,
#                jitter=1e-6,
#                white: bool = True) -> MeanAndVariance:
#     # TODO add noise???
#     Kmm = Kuu(X, kernel)

#     # Kmm += jitter * np.eye(Kmm.shape[0])
#     Kmn = kernel.K(X, Xnew)
#     if full_cov:
#         Knn = kernel.K(Xnew, Xnew)
#     else:
#         Knn = kernel.Kdiag(Xnew)

#     print('q_sqrt')
#     print(q_sqrt.shape)
#     # if q_sqrt is not None:
#     # TODO map over output dimension
#     # q_sqrt = np.squeeze(q_sqrt)
#     # q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

#     # TODO map over output dimension of Y??
#     # f += mean_func
#     fmean, fvar = base_conditional(Kmn=Kmn,
#                                    Kmm=Kmm,
#                                    Knn=Knn,
#                                    f=f,
#                                    full_cov=full_cov,
#                                    q_sqrt=q_sqrt,
#                                    white=white)
#     # return fmean, fvar
#     return fmean + mean_func, fvar


def gp_jacobian_hard_coded(cov_fn, Xnew, X, Y, jitter=1e-4):
    Xnew = Xnew.reshape(1, -1)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    Kxx = cov_fn(X, X)
    Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    # TODO check cholesky is implemented correctly
    kinvy = sp.linalg.solve_triangular(chol, Y, lower=True)

    # dk_dt0 = kernel.dK_dX(Xnew, X, 0)
    # dk_dt1 = kernel.dK_dX(Xnew, X, 1)
    # dk_dtT = np.stack([dk_dt0, dk_dt1], axis=1)
    # dk_dtT = np.squeeze(dk_dtT)
    dk_dtT = jacobian_cov_fn_wrt_x1(cov_fn, Xnew, X)

    v = sp.linalg.solve_triangular(chol, dk_dtT, lower=True)

    # TODO lengthscale shouldn't be hard codded
    lengthscale = np.array([0.4, 0.4])
    l2 = lengthscale**2
    # l2 = kernel.lengthscale**2
    l2 = np.diag(l2)
    d2k_dtt = -l2 * cov_fn(Xnew, Xnew)

    # calculate mean and variance of J
    # mu_j = np.dot(dk_dtT, kinvy)
    mu_j = v.T @ kinvy
    cov_j = d2k_dtt - np.matmul(v.T, v)  # d2Kd2t doesn't need to be calculated
    return mu_j, cov_j


def gp_jacobian(Xnew: InputData,
                X: InputData,
                kernels,
                mean_funcs: MeanFunc,
                f: OutputData,
                full_cov: bool = False,
                q_sqrt=None,
                jitter=1e-6,
                white: bool = True) -> MeanAndVariance:
    mu_js, cov_js = [], []
    print('inside gp_jacobian')
    print(mean_funcs)
    print(kernels)
    print(f.shape)
    print(X.shape)
    for output_dim, (kernel, mean_func) in enumerate(zip(kernels, mean_funcs)):
        if q_sqrt is not None:
            q_sqrt_ = q_sqrt[output_dim:output_dim + 1, :, :]
        else:
            q_sqrt_ = q_sqrt
        mu_j, cov_j = gp_jacobian_single_output(Xnew,
                                                X,
                                                kernel,
                                                mean_func,
                                                f[:,
                                                  output_dim:output_dim + 1],
                                                full_cov=full_cov,
                                                q_sqrt=q_sqrt_,
                                                jitter=jitter,
                                                white=white)
        mu_js.append(mu_j)
        cov_js.append(cov_j)
        print('mu_j')
        print(mu_j.shape)
        print(cov_j.shape)
    mu_js = np.stack(mu_j)
    cov_js = np.stack(cov_j)
    # print('inside gp_jacobian')
    print(output_dim)
    print(mu_js.shape)
    print(cov_js.shape)
    return mu_js, cov_js


def gp_jacobian_single_output(Xnew: InputData,
                              X: InputData,
                              kernel,
                              mean_func: MeanFunc,
                              f: OutputData,
                              full_cov: bool = False,
                              q_sqrt=None,
                              jitter=1e-6,
                              white: bool = True) -> MeanAndVariance:
    assert Xnew.shape[1] == X.shape[1]
    Kxx = kernel.K(X, X)
    Kxx += jitter * np.eye(Kxx.shape[0])
    print('Kxx')
    print(Kxx.shape)
    dKdx1 = jacobian_cov_fn_wrt_x1(kernel.K, Xnew, X)
    print('dKdx1')
    print(dKdx1.shape)
    # print(dKdx1)
    # d2K = hessian_cov_fn_wrt_x1x1(kernel.K, Xnew)
    # TODO cheating here - only works for RBF kernel
    lengthscale = kernel.lengthscale
    d2K = hessian_cov_fn_wrt_x1x1_hard_coded(kernel.K, lengthscale, Xnew)
    print('d2k')
    print(d2K.shape)
    # print(d2K)

    if q_sqrt is not None:
        # TODO map over output dimension
        # q_sqrt = np.squeeze(q_sqrt)
        q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

    mu_j, cov_j = base_conditional(Kmn=dKdx1,
                                   Kmm=Kxx,
                                   Knn=d2K,
                                   f=f,
                                   full_cov=full_cov,
                                   q_sqrt=q_sqrt,
                                   white=white)
    # TODO add derivative of mean_func - for constant mean_func this is zero
    return mu_j, cov_j


# def gp_jacobian(Xnew: InputData,
#                 X: InputData,
#                 kernel,
#                 mean_func: MeanFunc,
#                 f: OutputData,
#                 full_cov: bool = False,
#                 q_sqrt=None,
#                 jitter=1e-6,
#                 white: bool = True) -> MeanAndVariance:
#     assert Xnew.shape[1] == X.shape[1]
#     Kxx = kernel.K(X, X)
#     Kxx += jitter * np.eye(Kxx.shape[0])
#     dKdx1 = jacobian_cov_fn_wrt_x1(kernel.K, Xnew, X)
#     d2K = hessian_cov_fn_wrt_x1x1(kernel.K, Xnew)

#     if q_sqrt is not None:
#         # TODO map over output dimension
#         # q_sqrt = np.squeeze(q_sqrt)
#         q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

#     mu_j, cov_j = base_conditional(Kmn=dKdx1,
#                                    Kmm=Kxx,
#                                    Knn=d2K,
#                                    f=f,
#                                    full_cov=full_cov,
#                                    q_sqrt=q_sqrt,
#                                    white=white)
#     # TODO add derivative of mean_func - for constant mean_func this is zero
#     return mu_j, cov_j

# def gp_jacobian(cov_fn, Xnew, X, Y, jitter=1e-4):
#     print(Xnew.shape)
#     print(X.shape)
#     assert Xnew.shape[1] == X.shape[1]
#     Kxx = cov_fn(X, X)
#     Kxx += jitter * np.eye(Kxx.shape[0])
#     chol = sp.linalg.cholesky(Kxx, lower=True)
#     dKdx1 = jacobian_cov_fn_wrt_x1(cov_fn, Xnew, X)
#     d2K = hessian_cov_fn_wrt_x1x1(cov_fn, Xnew)

#     A1 = sp.linalg.solve_triangular(chol, dKdx1, lower=True)
#     A2 = sp.linalg.solve_triangular(chol, Y, lower=True)
#     ATA = A1.T @ A1

#     mu_j = A1.T @ A2
#     cov_j = d2K - ATA
#     # cov_j = 7 + d2K - ATA
#     # cov_j = 7 - ATA
#     return mu_j, cov_j

# def gp_predict(Xnew, X, Y, kernel, mean_func=0., jitter=1e-4):
# num_data = X.shape[0]

# Kxx = kernel.K(X, X)
# Kxx = Kxx + jitter * np.eye(Kxx.shape[0])
# chol = sp.linalg.cholesky(Kxx, lower=True)
# assert chol.shape == (num_data, num_data)
# kinvy = sp.linalg.solve_triangular(
#     chol.T, sp.linalg.solve_triangular(chol, Y, lower=True))
# assert kinvy.shape == (num_data, 1)

# # calculate mean and variance of J
# Kxs = kernel.K(X, x_star)
# mu = np.dot(Kxs.T, kinvy)

# Kss = kernel.K(x_star, x_star)
# v = sp.linalg.solve_triangular(chol, Kxs, lower=True)
# vT = v.T
# cov = Kss - np.matmul(vT, v)
# mu = mu + mean_func
# return mu, cov
