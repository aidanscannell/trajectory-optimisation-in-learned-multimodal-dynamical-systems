import abc

# from jax import jacfwd, jit, partial, vmap
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import objax
from gpjax.utilities import leading_transpose

from tromp.metric_tensors import RiemannianMetricTensor

States = jnp.ndarray


class ODE(objax.Module, abc.ABC):
    @abc.abstractmethod
    def ode_fn(self, times: jnp.ndarray, states: States) -> States:
        raise NotImplementedError


class GeodesicODE(ODE):
    def __init__(
        self, metric_tensor: RiemannianMetricTensor, jitter: jnp.float64 = None
    ):
        self.metric_tensor = metric_tensor
        if jitter is None:
            self.jitter = metric_tensor.jitter
        else:
            self.jitter = jitter

    def metric_fn(self, pos_guesses, full_cov: bool = True):
        # return self.metric_tensor(pos_guesses, full_cov=full_cov)
        return self.metric_tensor.metric_fn(pos_guesses, full_cov=full_cov)

    def ode_fn(self, times, states):
        print("inside ode_fn")
        print(states.shape)
        # jitter = 1e-6
        full_cov = True
        input_dim = int(states.shape[-1] / 2)
        if len(states.shape) == 1:
            num_states = 1
            # pos = states[:input_dim].reshape(1, input_dim)
            # vel = states[input_dim:].reshape(1, input_dim)
            pos = states[:input_dim].reshape(input_dim)
            vel = states[input_dim:].reshape(input_dim)
            kron_vel = jnp.kron(vel, vel).reshape(input_dim * 2, 1)
        elif len(states.shape) == 2:
            num_states = states.shape[0]
            pos = states[:, :input_dim].reshape(-1, input_dim)
            vel = states[:, input_dim:].reshape(-1, input_dim)
            kron_vel = jax.vmap(jnp.kron)(vel, vel)
            kron_vel = kron_vel[..., jnp.newaxis]
        print("pos")
        print(pos.shape)
        print(vel.shape)
        print("kron_vel")
        print(kron_vel.shape)

        # TODO implement value and jac
        grad_vec_metric_tensor_wrt_pos = (
            self.metric_tensor.grad_vec_metric_tensor_wrt_Xnew(
                Xnew=pos, full_cov=full_cov
            )
        )
        # grad_vec_metric_tensor_wrt_pos = grad_vec_metric_tensor_wrt_pos.reshape(
        #     input_dim, input_dim * input_dim)
        print("grad vec")
        print(grad_vec_metric_tensor_wrt_pos.shape)
        grad_vec_metric_tensor_wrt_posT = leading_transpose(
            grad_vec_metric_tensor_wrt_pos, [..., -1, -2]
        )
        print(grad_vec_metric_tensor_wrt_posT.shape)

        metric_tensor = self.metric_tensor(pos, full_cov)

        # TODO implement cholesky if metric_tensor is PSD
        # chol_lower = jsp.linalg.cholesky(metric_tensor, lower=True)
        # print("chol lower")
        # print(chol_lower)
        # inv_metric_tensor_grad_vec = jsp.linalg.solve_triangular(
        #     chol_lower.T,
        #     jsp.linalg.solve_triangular(
        #         chol_lower, grad_vec_metric_tensor_wrt_posT, lower=True
        #     ),
        # )
        # metric_tensor = metric_tensor + jnp.eye(input_dim) * self.jitter
        # metric_tensor = -metric_tensor + jnp.eye(input_dim) * 1e-4
        # metric_tensor = -metric_tensor
        # print("metric tensor")
        # print(metric_tensor)
        # c, low = jsp.linalg.cho_factor(metric_tensor, check_finite=True)
        # print("c")
        # print(c)
        # print(low)
        # inv_metric_tensor_grad_vec = jsp.linalg.cho_solve(
        #     (c, low), grad_vec_metric_tensor_wrt_posT
        # )
        # # # inv_metric_tensor_grad_vec = jsp.linalg.solve_triangular(
        # # #     chol_lower, grad_vec_metric_tensor_wrt_posT, lower=True
        # # # )
        # print("inv_metric_tensor_grad_vec")
        # print(inv_metric_tensor_grad_vec)

        # acc = -0.5 * inv_metric_tensor_grad_vec @ kron_vel

        metric_tensor = metric_tensor + jnp.eye(input_dim) * self.jitter
        inv_metric_tensor = jnp.linalg.inv(metric_tensor)
        print("inv")
        print(inv_metric_tensor)

        acc = (
            -0.5
            * inv_metric_tensor
            @ grad_vec_metric_tensor_wrt_posT
            @ kron_vel
        )
        print("acc")
        print(acc.shape)
        acc = acc.reshape(num_states, input_dim)
        if len(states.shape) == 1:
            vel = vel.reshape(1, input_dim)
        # elif len(states.shape) == 2:
        #     acc = acc.reshape(num_states, input_dim)
        # print(acc.shape)

        states_prime = jnp.concatenate([vel, acc], -1)
        print("states_prime")
        print(states_prime)

        # return flattended states_prime if states was flat
        if num_states == 1:
            return states_prime.reshape([-1])
        else:
            return states_prime

        # return state_prime
        # return state_prime.reshape([-1])
        # return state_prime.reshape(-1)


# class GeodesicODE(ODE):
#     def __init__(self, metric_fn, metric_fn_kwargs):
#         self.metric_fn = metric_fn
#         self.metric_fn_kwargs = metric_fn_kwargs

#     def ode_fn(self, time, state):
#         print("inside ode_fn")
#         print(state.shape)
#         input_dim = int(state.shape[0] / 2)
#         pos = state[:input_dim].reshape(1, -1)
#         vel = state[input_dim:].reshape(1, -1)
#         kron_vel = jnp.kron(vel, vel).T

#         # TODO implement value and jac
#         grad_func = jax.jacfwd(calc_vec_metric_tensor, 0)
#         grad_vec_metric_tensor_wrt_pos = grad_func(
#             pos.reshape(-1), self.metric_fn, self.metric_fn_kwargs
#         )
#         # grad_vec_metric_tensor_wrt_pos = grad_vec_metric_tensor_wrt_pos.reshape(
#         #     input_dim, input_dim * input_dim)
#         print("grad vec")
#         print(grad_vec_metric_tensor_wrt_pos.shape)
#         grad_vec_metric_tensor_wrt_posT = grad_vec_metric_tensor_wrt_pos.T

#         try:
#             metric_tensor, _ = self.metric_fn(pos, **self.metric_fn_kwargs)
#         except:
#             metric_tensor, _, _ = self.metric_fn(pos, **self.metric_fn_kwargs)
#         # TODO implement cholesky if metric_tensor is PSD
#         jitter = 1e-6
#         metric_tensor += jnp.eye(input_dim) * jitter
#         inv_metric_tensor = jnp.linalg.inv(metric_tensor)
#         print("inv")
#         print(inv_metric_tensor)
#         # assert jnp.isnan(inv_metric_tensor).any()

#         acc = (
#             -0.5
#             * inv_metric_tensor
#             @ grad_vec_metric_tensor_wrt_posT
#             @ kron_vel
#         )
#         acc = acc.reshape(1, input_dim)

#         state_prime = jnp.concatenate([vel, acc], -1)
#         print("state_prime")
#         print(state_prime)
#         # return state_prime
#         return state_prime.reshape([-1])

#     @jax.partial(jax.jit, static_argnums=(1, 2))
#     def calc_vec_metric_tensor(pos, metric_fn, metric_fn_kwargs):
#         print("here calc vec metric tensor")
#         print(pos.shape)
#         pos = pos.reshape(1, -1)
#         try:
#             metric_tensor, _ = metric_fn(pos, **metric_fn_kwargs)
#             # TODO add the correct exception
#         except:
#             metric_tensor, _, _ = metric_fn(pos, **metric_fn_kwargs)
#         input_dim = pos.shape[1]
#         vec_metric_tensor = metric_tensor.reshape(
#             input_dim * input_dim,
#         )
#         return vec_metric_tensor
