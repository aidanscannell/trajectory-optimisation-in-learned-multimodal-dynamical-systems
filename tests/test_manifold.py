import chex
import jax
from absl.testing import parameterized
from gpjax.kernels import SeparateIndependent, SquaredExponential
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from gpjax.models import SVGP
from jax import numpy as jnp
from tromp.manifold import GPManifold

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

# SVGP variants
num_inducing = 30
input_dim = 2
output_dim = 1
if output_dim > 1:
    kernels = [
        SquaredExponential(
            lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
        )
        for _ in range(output_dim)
    ]
    kernel = SeparateIndependent(kernels)
else:
    kernel = SquaredExponential(
        lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
    )
inducing_variable = jax.random.uniform(key=key, shape=(num_inducing, input_dim))

svgp = SVGP(
    kernel,
    likelihood=Gaussian(),
    inducing_variable=inducing_variable,
    mean_function=Constant(output_dim=output_dim),
    num_latent_gps=output_dim,
    q_diag=False,
    whiten=False,
)

params = svgp.get_params()


# Data variants
num_datas = [None, 1, 300]
manifold = GPManifold(gp=svgp)


class TestGPManifold(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(num_data=num_datas)
    def test_inner_product(self, num_data):
        """Check shapes of output"""
        if num_data is not None:
            u = jnp.ones([num_data, input_dim])
            v = jnp.ones([num_data, input_dim])
            Xnew = jax.random.uniform(key, shape=(num_data, input_dim))
        else:
            u = jnp.ones([input_dim])
            v = jnp.ones([input_dim])
            Xnew = jax.random.uniform(key, shape=(input_dim,))

        def inner_product(Xnew, u, v):
            return manifold.inner_product(Xnew, u, v)

        var_inner_product = self.variant(inner_product)
        inner_prod = var_inner_product(Xnew, u, v)
        if num_data is not None:
            assert inner_prod.shape[0] == num_data
        else:
            assert inner_prod.shape == ()

        def geodesic_ode(pos, vel):
            return manifold.geodesic_ode(pos, vel)

        var_geodesic_ode = self.variant(geodesic_ode)
        state_prime = var_geodesic_ode(Xnew, u)
        if num_data is None:
            assert state_prime.ndim == 1
            assert state_prime.shape[0] == 2 * input_dim
        else:
            assert state_prime.ndim == 2
            assert state_prime.shape[0] == num_data
            assert state_prime.shape[1] == 2 * input_dim
