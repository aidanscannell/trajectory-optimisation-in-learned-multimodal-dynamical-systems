import abc

import jax
import jax.numpy as jnp
import objax
import scipy as sp
from bunch import Bunch
from jax.config import config
from scipy.optimize import Bounds, NonlinearConstraint

from tromp.metric_tensors import RiemannianMetricTensor
# from tromp.ode import geodesic_ode
from tromp.ode import ODE, GeodesicODE

config.update("jax_enable_x64", True)


class BaseSolver(objax.Module, abc.ABC):
    def __init__(self, ode: ODE):
        self.ode = ode
        # self.times = times

    @abc.abstractmethod
    def objective_fn(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def solve_trajectory(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
    ):
        raise NotImplementedError


def start_end_pos_bounds(state_guesses, pos_init, pos_end):
    lb = -jnp.ones([*state_guesses.shape]) * jnp.inf
    ub = jnp.ones([*state_guesses.shape]) * jnp.inf

    for idx, pos in enumerate(pos_init):
        if pos < 0:
            lb = jax.ops.index_update(lb, jax.ops.index[0, idx], pos * 1.02)
            ub = jax.ops.index_update(ub, jax.ops.index[0, idx], pos * 0.98)
        else:
            lb = jax.ops.index_update(lb, jax.ops.index[0, idx], pos * 0.98)
            ub = jax.ops.index_update(ub, jax.ops.index[0, idx], pos * 1.02)

    for idx, pos in enumerate(pos_end):
        if pos < 0:
            lb = jax.ops.index_update(lb, jax.ops.index[-1, idx], pos * 1.02)
            ub = jax.ops.index_update(ub, jax.ops.index[-1, idx], pos * 0.98)
        else:
            lb = jax.ops.index_update(lb, jax.ops.index[-1, idx], pos * 0.98)
            ub = jax.ops.index_update(ub, jax.ops.index[-1, idx], pos * 1.02)

    bounds = Bounds(lb=lb.flatten(), ub=ub.flatten())
    return bounds


class GeodesicSolver(objax.Module, abc.ABC):
    def __init__(self, ode: GeodesicODE):
        self.ode = ode
        # self.times = times

    @abc.abstractmethod
    def objective_fn(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def solve_trajectory(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
    ):
        raise NotImplementedError


class CollocationGeodesicSolver(BaseSolver):
    def __init__(
        self,
        ode,
        # num_col_points: int = 10,
        covariance_weight: jnp.float64 = 1.0,
        maxiter: int = 100,
    ):
        super().__init__(ode)
        self.covariance_weight = covariance_weight
        self.maxiter = maxiter

    def collocation_constraint_fn(self, state_guesses):
        times = self.times  # remove this
        input_dim = 2
        state_guesses = state_guesses.reshape([-1, 2 * input_dim])
        num_timesteps = times.shape[0]
        dt = times[-1] - times[0]
        time_col = jnp.linspace(
            times[0] + dt / 2, times[-1] - dt / 2, num_timesteps - 1
        )

        def ode_fn(state):
            return self.ode.ode_fn(times, state)

        state_prime = jax.vmap(ode_fn)(state_guesses)
        # state_prime = ode_fn(state_guesses)
        state_ll = state_guesses[0:-1, :]
        state_rr = state_guesses[1:, :]
        state_prime_ll = state_prime[0:-1, :]
        state_prime_rr = state_prime[1:, :]
        state_col = 0.5 * (state_ll + state_rr) + dt / 8 * (
            state_prime_ll - state_prime_rr
        )
        state_prime_col = jax.vmap(ode_fn)(state_col)
        # state_prime_col = ode_fn(state_col)

        defect = (state_ll - state_rr) + dt / 6 * (
            state_prime_ll + 4 * state_prime_col + state_prime_rr
        )
        print("defect")
        print(defect)
        return defect.flatten()

    # @jax.partial(jax.jit, static_argnums=(1, 2, 3))
    # @jax.partial(objax.Jit, static_argnums=(1, 2, 3))
    def objective_fn(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
    ):
        input_dim = 2
        state_guesses = state_guesses.reshape([-1, 2 * input_dim])
        pos_guesses = state_guesses[:, 0:2]
        pos_guesses = jax.ops.index_update(
            pos_guesses, jax.ops.index[0, :], pos_init
        )
        pos_guesses = jax.ops.index_update(
            pos_guesses, jax.ops.index[-1, :], pos_end_targ
        )
        norm = jnp.linalg.norm(pos_guesses, axis=-1, ord=-2)

        metric_tensor = self.ode.metric_fn(pos_guesses)
        # print("metric_tensor")
        # print(metric_tensor.shape)
        # metric_tensor, jac = metric_fn(pos_guesses, **metric_fn_kwargs)
        trace_metric = jnp.trace(metric_tensor, axis1=-2, axis2=-1)
        # print(trace_metric.shape)
        trace_metric_sum = jnp.sum(trace_metric)

        norm_sum = jnp.sum(norm)
        print("Norm Loss: ", norm_sum)
        print("Trace Metric Loss: ", trace_metric_sum)
        # print("Time Loss: ", times[-1])
        # metric_weight = 5.0
        # metric_weight = 50.0
        # return norm_sum
        # return norm_sum + self.covariance_weight * trace_metric_sum
        # return trace_metric_sum
        return 1.0

    def solve_trajectory(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
        bounds: Bounds = None,
    ):
        if bounds is None:
            bounds = start_end_pos_bounds(
                state_guesses, pos_init, pos_end_targ
            )

        states_shape = state_guesses.shape
        # state_dim = states_guesses.shape[1]
        state_guesses = state_guesses.reshape(-1)
        # state_guesses_var = objax.TrainVar(state_guesses)
        state_guesses_var = objax.StateVar(state_guesses)

        objective_args = (pos_init, pos_end_targ, times)
        self.times = times  # TODO delete this!!

        # defect_constraints = NonlinearConstraint(
        #     self.collocation_constraint_fn, -0.1, 0.1
        # )
        constraint_fn_vars = {"state_guesses": state_guesses_var}
        # constraint_fn_vars = state_guesses_var
        constraint_fn_vars = objax.VarCollection(constraint_fn_vars)
        jitted_constraint_fn = objax.Jit(
            self.collocation_constraint_fn, constraint_fn_vars
        )
        defect_constraints = NonlinearConstraint(
            # jitted_constraint_fn,
            # -0.1,
            # 0.1
            jitted_constraint_fn,
            -0.001,
            0.001,
        )

        jitted_objective_fn = objax.Jit(self.objective_fn, constraint_fn_vars)

        # defect_constraints = NonlinearConstraint(
        #     self.collocation_constraint_fn, -0.01, 0.01
        # )

        constraints = defect_constraints
        method = "SLSQP"

        res = sp.optimize.minimize(
            # self.objective_fn,
            jitted_objective_fn,
            # jnp.flatten(state_guesses),
            state_guesses,
            # params,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"verbose": 1, "disp": True, "maxiter": self.maxiter},
            args=objective_args,
        )
        print("res")
        print(res)
        print(res.x.shape)
        state_opt = res.x
        # state_opt = res.x[:-1]
        # state_opt = state_opt.reshape([*state_guesses.shape])
        state_opt = state_opt.reshape(states_shape)
        return state_opt


class HermiteSimpsonCollocationSolver(CollocationGeodesicSolver):
    def __init__(self, ode, num_col_points: int = 10):
        super().__init__(ode)

    def collocation_constraints_fn():
        return 0
