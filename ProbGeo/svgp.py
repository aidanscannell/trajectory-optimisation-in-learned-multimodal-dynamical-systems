from jax import numpy as np
from jax import scipy as sp
from jax import partial, jit, jacfwd, jacrev, vmap
# from typing import Bool

from ProbGeo.covariances import Kuu, Kuf, jacobian_cov_fn_wrt_x1, hessian_cov_fn_wrt_x1x1
from ProbGeo.conditionals import base_conditional


def sparse_gp_predict(Xnew,
                      Z,
                      kernel,
                      mean_func,
                      q_mu,
                      full_cov=False,
                      q_sqrt=None,
                      jitter=1e-8,
                      white: bool = True):
    Kmm = Kuu(Z, kernel)
    Kmn = kernel.K(Z, Xnew)
    if full_cov:
        Knn = kernel.K(Xnew, Xnew)
    else:
        Knn = kernel.Kdiag(Xnew)

    # TODO map over output dimension
    q_sqrt = np.squeeze(q_sqrt)
    fmean, fvar = base_conditional(Kmn,
                                   Kmm,
                                   Knn,
                                   f=q_mu,
                                   full_cov=full_cov,
                                   q_sqrt=q_sqrt,
                                   white=white)
    return fmean + mean_func, fvar


def sparse_gp_jacobian(Xnew,
                       Z,
                       kernel,
                       mean_func,
                       q_mu,
                       full_cov=False,
                       q_sqrt=None,
                       jitter=1e-8,
                       white: bool = True):

    # TODO what to do with mean_func???
    # TODO handle output dimension > 1
    q_sqrt = np.squeeze(q_sqrt)

    def single_sparse_gp_jacobian(x):
        x = x.reshape(1, -1)
        cov_fn = kernel.K
        d2K = hessian_cov_fn_wrt_x1x1(cov_fn, x)
        dK = jacobian_cov_fn_wrt_x1(cov_fn, x, Z)
        Kmm = Kuu(Z, kernel)
        mu_j, cov_j = base_conditional(Kmn=dK,
                                       Kmm=Kmm,
                                       Knn=d2K,
                                       f=q_mu,
                                       full_cov=full_cov,
                                       q_sqrt=q_sqrt,
                                       white=white)
        # TODO Add jacobian mean (if gp mean func is not just constant)
        return mu_j, cov_j

    return vmap(single_sparse_gp_jacobian, in_axes=(0))(Xnew)
