from jax import batching, jacfwd, jit, jvp, lu
from jax import numpy as np
from jax import partial, tree_map, vmap

from sparse_probabilistic_metric import calc_G_map_sparse
from probabilistic_geodesic import value_and_jacfwd


####################################
# sparse GPs
####################################
@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def calc_vecG_sparse(c, X, z, q_mu, q_sqrt, kernel, mean_func, m_h_mu):
    G, _, _ = calc_G_map_sparse(c, X, z, q_mu, q_sqrt, kernel, mean_func,
                                m_h_mu)
    input_dim = X.shape[1]
    vecG = G.reshape(input_dim * input_dim, )  # order doesnt matter as diag
    return vecG


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def geodesic_fun_sparse(c, g, X, z, q_mu, q_sqrt, kernel, mean_func, m_h_mu):
    if len(c.shape) < 2:
        c = c.reshape(1, c.shape[0])
    if len(g.shape) < 2:
        g = g.reshape(1, g.shape[0])
    kronC = np.kron(g, g).T

    # this works with my new JAX function
    val_grad_func = value_and_jacfwd(calc_vecG_sparse, 0)
    vecG, dvecGdc = val_grad_func(c, X, z, q_mu, q_sqrt, kernel, mean_func,
                                  m_h_mu)
    G = vecG.reshape(c.shape[1], c.shape[1])

    invG = np.linalg.inv(G)
    dvecGdc = dvecGdc[:, 0, :].T
    return -0.5 * invG @ dvecGdc @ kronC
