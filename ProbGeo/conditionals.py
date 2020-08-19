from jax import numpy as np
from jax import scipy as sp


def base_conditional(Kmn,
                     Kmm,
                     Knn,
                     f,
                     *,
                     full_cov=False,
                     q_sqrt=None,
                     white=False):
    Lm = sp.linalg.cholesky(Kmm, lower=True)
    return base_conditional_with_lm(Kmn=Kmn,
                                    Lm=Lm,
                                    Knn=Knn,
                                    f=f,
                                    full_cov=full_cov,
                                    q_sqrt=q_sqrt,
                                    white=white)


def base_conditional_with_lm(Kmn,
                             Lm,
                             Knn,
                             f,
                             *,
                             full_cov=False,
                             q_sqrt=None,
                             white=False):
    A = sp.linalg.solve_triangular(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - A.T @ A
    else:
        fvar = Knn - np.sum(np.square(A), -2)

    # another backsubstitution in the unwhitened case
    if not white:
        A = sp.linalg.solve_triangular(Lm.T, A, lower=False)

    # construct the conditional mean
    fmean = A.T @ f

    if q_sqrt is not None:
        q_sqrt_dims = len(q_sqrt.shape)
        if q_sqrt_dims == 2:
            # LTA = sp.linalg.solve_triangular(Lm, q_sqrt)
            LTA = q_sqrt.T @ A
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt_dims))

        if full_cov:
            # fvar = fvar + LTA @ LTA.T
            fvar = fvar + LTA.T @ LTA
        else:
            fvar = fvar + np.sum(np.square(LTA), -2)

        return fmean, fvar
