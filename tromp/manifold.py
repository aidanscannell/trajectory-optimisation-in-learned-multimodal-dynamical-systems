#!/usr/bin/env python3
import abc
import typing
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import tensor_annotations.jax as tjax
from gpjax.custom_types import InputDim, MeanAndCovariance, NumData, OutputDim
from gpjax.models import GPModel
from jax import grad
from tensor_annotations import axes

TwoInputDim = typing.NewType("TwoInputDim", axes.Axis)  # 2*InputDim
# from tromp.curve import BasicCurve


class Manifold(abc.ABC):
    def curve_length(self, curve):
        """Compute the discrete length of a given curve."""
        raise NotImplementedError

    @abc.abstractmethod
    def _metric(self, point: tjax.Array1[InputDim]) -> tjax.Array2[InputDim, InputDim]:
        """Return the metric tensor at a specified point."""
        raise NotImplementedError

    @jax.partial(jnp.vectorize, excluded=(0,), signature="(d)->(d,d)")
    def metric(
        self, points: Union[tjax.Array1[InputDim], tjax.Array2[NumData, InputDim]]
    ) -> Union[
        tjax.Array2[InputDim, InputDim], tjax.Array3[NumData, InputDim, InputDim]
    ]:
        """Return the metric tensor at a specified SET of points."""
        return self._metric(points)

    def _inner_product_given_metric(
        self,
        metric: tjax.Array2[InputDim, InputDim],
        u: tjax.Array1[InputDim],
        v: tjax.Array1[InputDim],
    ) -> jnp.float64:
        """Compute inner product between tangent vectors u and v at base point.

        :param metric: metric tensor at base point
        :param u: tangent vector in tangent space of base
        :param v: tangent vector in tangent space of base
        :returns: inner product between u and v according to metric @ base
        """
        return u.T @ metric @ v

    @jax.partial(jnp.vectorize, excluded=(0,), signature="(d),(d),(d)->()")
    def inner_product_given_metric(self, metric, U, V):
        """Compute inner product between sets of tangent vectors u and v at base points.

        :param metric: metric tensor at base points [num_data, input_dim, input_dim]
        :param U: set of tangent vectors in tangent space of base [num_data, input_dim]
        :param V: set of tangent vectors in tangent space of base [num_data, input_dim]
        :returns: inner product between u and v according to metric @ base [num_data]
        """
        return self._inner_product_given_metric(metric, U, V)

    @jax.partial(jnp.vectorize, excluded=(0,), signature="(d),(d),(d)->()")
    def inner_product(self, base, U, V):
        """Compute inner product between sets of tangent vectors u and v at base points.

        :param base: base points [num_data, input_dim, input_dim]
        :param U: set of tangent vectors in tangent space of base [num_data, input_dim]
        :param V: set of tangent vectors in tangent space of base [num_data, input_dim]
        :returns: inner product between u and v according to metric @ base [num_data]
        """
        metric = self._metric(base)
        return self._inner_product_given_metric(metric, U, V)

    def _geodesic_ode(
        self, pos: tjax.Array1[InputDim], vel: tjax.Array1[InputDim]
    ) -> tjax.Array1[TwoInputDim]:
        """Evaluate the geodesic ODE of the manifold."""
        vec_metric = lambda x: jnp.reshape(self.metric(x), -1)
        grad_vec_metric_wrt_pos = jax.jacfwd(vec_metric)(pos)
        metric = self.metric(pos)
        inner_prod = self._inner_product_given_metric(metric, vel, vel)

        kron_vel = jnp.kron(vel, vel).T

        jitter = 1e-4
        pos_dim = pos.shape[0]
        metric += jnp.eye(pos_dim) * jitter
        # rhs = grad_vec_metric_wrt_pos.T @ kron_vel
        # chol = jsp.linalg.cholesky(metric, lower=True)
        # inv_metric = jsp.linalg.solve_triangular(chol, rhs, lower=True)
        # print(inv_metric)
        # acc = -0.5 * inv_metric
        # print(acc)

        inv_metric = jnp.linalg.inv(metric)
        acc = -0.5 * inv_metric @ grad_vec_metric_wrt_pos.T @ kron_vel

        state_prime = jnp.concatenate([vel, acc])
        return state_prime

    @jax.partial(jnp.vectorize, excluded=(0,), signature="(d),(d)->(2d)")
    def geodesic_ode(
        self,
        pos: Union[tjax.Array1[InputDim], tjax.Array2[NumData, InputDim]],
        vel: Union[tjax.Array1[InputDim], tjax.Array2[NumData, InputDim]],
    ) -> Union[tjax.Array1[TwoInputDim], tjax.Array2[NumData, TwoInputDim]]:
        """Evaluate the geodesic ODE of the manifold.

        :param pos: array of points in the input space
        :param vel: array representing the velocities at the points
        :returns: array of accelerations at the points
        """
        return self._geodesic_ode(pos, vel)


class GPManifold(Manifold):
    """
    A common interface for embedded manifolds. Specific embedded manifolds
    should inherit from this abstract base class abstraction.
    """

    def __init__(
        self,
        gp: GPModel,
        covariance_weight: Optional[jnp.float64] = 1.0,
        gp_params: Optional[dict] = None,
    ):
        self.gp = gp
        self.covariance_weight = covariance_weight
        if gp_params is None:
            self.gp_params = gp.get_params()
        else:
            self.gp_params = gp_params

    def _metric(self, point: tjax.Array1[InputDim]) -> tjax.Array2[InputDim, InputDim]:
        """Return the metric tensor at a specified point."""
        jac_mean, jac_var = self._embed_jac(point)
        M = jac_mean.T @ jac_mean + self.covariance_weight * jac_var
        return M

    def _embed(
        self, point: tjax.Array1[InputDim]
    ) -> Tuple[tjax.Array1[OutputDim], tjax.Array1[OutputDim]]:
        """Embed the point into (mu, var) space."""
        means, vars = self.gp.predict_f(self.gp_params, point, full_cov=False)
        return means.reshape(-1), vars.reshape(-1)

    @jax.partial(jnp.vectorize, excluded=(0,), signature="(d)->(p),(p)")
    def embed(
        self, points: Union[tjax.Array1[InputDim], tjax.Array2[NumData, InputDim]]
    ) -> MeanAndCovariance:
        """Embed the manifold into (mu, var) space."""
        return self._embed(points)

    def _embed_jac(self, point):
        return jax.jacfwd(self._embed, -1)(point)

    @jax.partial(jnp.vectorize, excluded=(0,), signature="(d)->(p,d),(p,d)")
    def embed_jac(self, points):
        return self._embed_jac(points)
