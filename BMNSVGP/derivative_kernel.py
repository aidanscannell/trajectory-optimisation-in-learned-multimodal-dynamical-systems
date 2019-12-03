import gpflow
import numpy as np
import tensorflow as tf
from gpflow.decors import params_as_tensors
from gpflow.kernels import SquaredExponential


class SquaredExponentialDerivative(SquaredExponential):
    """
    Squared exponential kernel extended to have derivatives wrt inputs
    """
    @params_as_tensors
    def K_r2(self, r2):
        return self.variance * tf.exp(-r2 / 2.)

    def dKdX(self, X, X2):
        """
        :parameters:
        X: [NxD]
        X2: [1xD]
        """
        assert X.shape[1] == X2.shape[1]
        assert X2.shape[0] == 1
        assert len(X2.shape) == 2
        print('using derivative dK()')
        k = (X - X2) * self.K(X, X2)  # [[NxD - 1xD] * Nx1]
        l = tf.expand_dims(self.lengthscales.parameter_tensor, 0)  # [1xD]
        k = k / l**2
        return k  # [NxD]

    def d2Kd2X(self, X, X2):
        print('using derivative d2K()')
        k = self.K(X, X2)
        d2k = tf.diag(self.lengthscales.parameter_tensor**2) * k
        return d2k

    # def d2Kd2X(self, X, X2):
    #     # TODO: only need upper triangle of Hessian (due to symmetry)
    #     print('using derivative d2K()')
    #     diff = X - X2  # [MxD]
    #     # TODO: should one of the diffs be transposed??
    #     diff_i = tf.expand_dims(diff, -1)  # [MxDx1]
    #     diff_j = tf.expand_dims(diff, -2)  # [Mx1xD]
    #     diff_ij = diff_i @ diff_j  # [MxDxD]
    #     K = self.K(X, X2)  # [1x1]
    #     l = tf.expand_dims(self.lengthscales.parameter_tensor, 0)  # [1xD]
    #     d2k = diff_ij / l  # [MxDxD] TODO: how should lengthscales be broadcast??
    #     d2k = tf.squeeze(d2k)  # TODO: should I squeeze here??

    #     d2k_diag = (diff**2 / l - 1
    #                 ) / l  # [1xD] TODO: how should lengthscales be broadcast??

    #     d2k = tf.linalg.set_diag(d2k, tf.squeeze(d2k_diag))

    #     d2k = d2k * self.K(X, X2)
    #     return d2k
