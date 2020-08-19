import jax.numpy as np
from jax import jit, partial


class DiffRBF:
    """
    Squared exponential kernel extended to have derivatives wrt inputs
    """

    # @partial(jit, static_argnums=(0, 1, 2, 3, 4))
    def __init__(self, input_dim, variance, lengthscale, ARD=False):
        self.ARD = ARD
        # if not ARD:
        #     if lengthscale is None:
        #         lengthscale = np.ones(1)
        #     else:
        #         lengthscale = np.asarray(lengthscale)
        #         assert lengthscale.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        # else:
        #     if lengthscale is not None:
        #         lengthscale = np.asarray(lengthscale)
        #         assert lengthscale.size in [1, input_dim
        #                                     ], "Bad number of lengthscales"
        #         if lengthscale.size != input_dim:
        #             lengthscale = np.ones(input_dim) * lengthscale
        #     else:
        #         lengthscale = np.ones(self.input_dim)
        lengthscale = np.asarray(lengthscale)
        self.lengthscale = lengthscale
        self.variance = variance
        # self.X = X
        # self.dimX = dimX
        # assert self.variance.size == 1

    @partial(jit, static_argnums=(0, ))
    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    @partial(jit, static_argnums=(0, ))
    def K(self, X, X2=None):
        """
        Kernel function applied on inputs X and X2.
        In the stationary case there is an inner function depending on the
        distances from X to X2, called r.
        K(X, X2) = K_of_r((X-X2)**2)
        """
        r = self._scaled_dist(X, X2)
        return self.K_of_r(r)

    def Kdiag(self, X):
        ret = np.ones(X.shape[0]) * self.variance
        return ret

    # def dK_dX(self, X2):
    #     r = self._scaled_dist(X, X2)
    #     K = self.K_of_r(r)
    #     dist = X[:, None, dimX] - X2[None, :, dimX]
    #     lengthscale2inv = (np.ones((X.shape[1])) / (self.lengthscale**2))[dimX]
    #     return -1. * K * dist * lengthscale2inv

    # @partial(jit, static_argnums=(0, 2))
    def dK_dX(self, X, X2, dimX):
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)
        dist = X[:, None, dimX] - X2[None, :, dimX]
        lengthscale2inv = (np.ones((X.shape[1])) / (self.lengthscale**2))[dimX]
        return -1. * K * dist * lengthscale2inv

    @partial(jit, static_argnums=(0, ))
    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        #X, = self._slice_X(X)
        # if X2 == None:
        #     Xsq = np.sum(np.square(X), 1)
        #     r2 = -2. * tdot(X) + (Xsq[:, None] + Xsq[None, :])
        #     util.diag.view(
        #         r2
        #     )[:,
        #       ] = 0.  # force diagnoal to be zero: sometime numerically a little negative
        #     # r2 = np.clip(r2, 0, np.inf)
        #     return np.sqrt(r2)
        # else:
        #X2, = self._slice_X(X2)
        X1sq = np.sum(np.square(X), 1)
        X2sq = np.sum(np.square(X2), 1)
        r2 = -2. * np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
        r2 = np.clip(r2, 0, np.inf)
        # r2 = np.abs(r2) + 0.0000007
        r2 += 0.0000007
        return np.sqrt(r2)

    @partial(jit, static_argnums=(0, ))
    def _scaled_dist(self, X, X2=None):
        if self.ARD:
            # if X2 is not None:
            X2 = X2 / self.lengthscale
            return self._unscaled_dist(X / self.lengthscale, X2)
        else:
            return self._unscaled_dist(X, X2) / self.lengthscale

    @partial(jit, static_argnums=(0, ))
    def K_r2(self, r2):
        return self.variance * np.exp(-r2 / 2.)

    # def dK_dX(self, X, X2):
    #     """
    #     :parameters:
    #     X: [NxD]
    #     X2: [1xD]
    #     """
    #     # r = self._scaled_dist(X, X2)
    #     # K = self.K_of_r(r)
    #     # dist = X[:,None,dimX]-X2[None,:,dimX]
    #     # lengthscale2inv = (np.ones((X.shape[1]))/(self.lengthscale**2))[dimX]
    #     # return -1.*K*dist*lengthscale2inv

    #     # assert X.shape[1] == X2.shape[1]
    #     # assert X2.shape[0] == 1
    #     # assert len(X2.shape) == 2
    #     print('using derivative dK()')
    #     k = (X - X2) * self.K(X, X2)  # [[NxD - 1xD] * Nx1]
    #     l = np.expand_dims(self.lengthscales.parameter_tensor, 0)  # [1xD]
    #     k = k / l**2
    #     return k  # [NxD]

    # def d2Kd2X(self, X, X2):
    #     print('using derivative d2K()')
    #     k = self.K(X, X2)
    #     d2k = tf.diag(self.lengthscales.parameter_tensor**2) * k
    #     return d2k


if __name__ == "__main__":
    from jax import grad, jacrev
    params = np.load('saved_models/params.npz')
    l = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    y = params['a']  # [num_data x 2] meen and var of alpha
    Y = y[0:1, :, 0].T
    dimX = 0
    kern = DiffRBF(2, X, dimX, variance=var, lengthscale=l, ARD=True)
    print(X.shape)

    t = np.ones([1, 2]) * 0.3
    dg = kern.dK_dX(t)
    print(dg.shape)
    dg = kern.dK_dX

    # dgdt = grad(dg)
    dgdt = jacrev(dg)
    print(dgdt(t).shape)
