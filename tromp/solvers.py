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


def state_guesses_to_opt_vars(state_guesses):
    # Flatten state guesses
    state_dim = state_guesses.shape[1]
    pos_dim = int(state_dim / 2)
    num_states = state_guesses.shape[0]
    state_guesses = state_guesses.reshape(-1)

    # Remove start state from optimisation variables
    state_guesses = state_guesses[pos_dim:]
    print("state guesses removed start pos")
    print(state_guesses.shape)
    # Remove end pos from optimisation variables
    # state_guesses = state_guesses[:-state_dim]
    # end_state_start_idx = num_states * state_dim - state_dim - pos_dim
    # end_pos_idxs = jnp.arange(end_state_start_idx,end_state_start_idx+pos_dim)
    # print("end_pos_idxs")
    # print(end_pos_idxs)
    # state_guesses = jnp.delete(state_guesses, end_pos_idxs)
    state_guesses_before = state_guesses[:-state_dim]
    print("state_guesses_before")
    print(state_guesses_before.shape)
    state_guesses_end = state_guesses[-state_dim + pos_dim :]
    print("state_guesses_end")
    print(state_guesses_end.shape)
    state_guesses = jnp.concatenate(
        [state_guesses_before, state_guesses_end], axis=0
    )
    print("state guesses removed end pos")
    print(state_guesses.shape)
    return state_guesses


# def start_end_pos_bounds(state_guesses, pos_init, pos_end):
#     lb = -jnp.ones([*state_guesses.shape]) * jnp.inf
#     ub = jnp.ones([*state_guesses.shape]) * jnp.inf

#     for idx, pos in enumerate(pos_init):
#         if pos < 0:
#             lb = jax.ops.index_update(lb, jax.ops.index[0, idx], pos * 1.02)
#             ub = jax.ops.index_update(ub, jax.ops.index[0, idx], pos * 0.98)
#         else:
#             lb = jax.ops.index_update(lb, jax.ops.index[0, idx], pos * 0.98)
#             ub = jax.ops.index_update(ub, jax.ops.index[0, idx], pos * 1.02)

#     for idx, pos in enumerate(pos_end):
#         if pos < 0:
#             lb = jax.ops.index_update(lb, jax.ops.index[-1, idx], pos * 1.02)
#             ub = jax.ops.index_update(ub, jax.ops.index[-1, idx], pos * 0.98)
#         else:
#             lb = jax.ops.index_update(lb, jax.ops.index[-1, idx], pos * 0.98)
#             ub = jax.ops.index_update(ub, jax.ops.index[-1, idx], pos * 1.02)

#     bounds = Bounds(lb=lb.flatten(), ub=ub.flatten())
#     return bounds


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

    def collocation_defects(self, opt_vars):
        times = self.times  # remove this
        # num_timesteps = times.shape[0]
        num_states = times.shape[0]
        state_guesses = self.opt_vars_to_states(
            opt_vars, self.pos_init, self.pos_end_targ, num_states
        )
        # state_dim = 4
        print("inside defects")
        print(state_guesses.shape)
        # state_guesses = state_guesses.reshape([-1, state_dim])
        dt = times[-1] - times[0]
        time_col = jnp.linspace(
            times[0] + dt / 2, times[-1] - dt / 2, num_states - 1
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

    def dummy_objective_fn(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
    ):
        return 1.0

    def objective_fn(
        self,
        opt_vars,
        pos_init,
        pos_end_targ,
        times,
    ):
        print("inside objective")
        if len(pos_init.shape) == 1:
            input_dim = pos_init.shape[0]
        num_states = times.shape[0]
        state_guesses = self.opt_vars_to_states(
            opt_vars, pos_init, pos_end_targ, num_states
        )
        print(state_guesses.shape)

        state_guesses = state_guesses.reshape([-1, 2 * input_dim])
        pos_guesses = state_guesses[:, 0:input_dim]

        # Calculate Euclidean distance
        # norm = jnp.linalg.norm(pos_guesses, axis=-1, ord=-2)
        norm = jnp.linalg.norm(pos_guesses, axis=-1)
        # norm = jnp.linalg.norm(pos_guesses[:, 0] - pos_guesses[:, 1])
        print("norm")
        print(norm.shape)
        norm_sum = jnp.sum(norm)
        print(norm_sum.shape)
        print("Norm Loss: ", norm_sum)
        # dist = sp.spatial.distance.cdist(pos_guesses[:,0],pos_guesses[:,1], metric="euclidean")

        # Calculate sum of metric trace along trajectory
        metric_tensor = self.ode.metric_fn(pos_guesses)
        trace_metric = jnp.trace(metric_tensor, axis1=-2, axis2=-1)
        trace_metric_sum = jnp.sum(trace_metric)
        print("Trace Metric Loss: ", trace_metric_sum)

        return norm_sum + self.covariance_weight * trace_metric_sum

    def sum_of_squares_objective(
        self,
        opt_vars,
        pos_init,
        pos_end_targ,
        times,
    ):
        if len(pos_init.shape) == 1:
            pos_dim = pos_init.shape[0]
        num_states = times.shape[0]
        state_guesses = self.opt_vars_to_states(
            opt_vars, pos_init, pos_end_targ, num_states
        )
        pos_guesses = state_guesses[:, :pos_dim]
        sum_of_squares = jnp.sum(pos_guesses ** 2)
        # sum_of_squares = jnp.sum(state_guesses ** 2)
        return sum_of_squares * 1000

    def opt_vars_to_states(self, opt_vars, pos_init, pos_end_targ, num_states):
        if len(pos_init.shape) == 1:
            pos_dim = pos_init.shape[0]
            state_dim = 2 * pos_dim
        state_guesses = opt_vars[: num_states * state_dim - 2 * pos_dim]
        # Add start pos
        state_guesses = jnp.concatenate([pos_init, state_guesses], axis=0)

        # Split state_guesses and insert end pos
        state_guesses_before = state_guesses[:-pos_dim]
        vel_end = state_guesses[-pos_dim:]
        state_guesses = jnp.concatenate(
            [state_guesses_before, pos_end_targ], axis=0
        )
        state_guesses = jnp.concatenate([state_guesses, vel_end], axis=0)

        state_guesses = state_guesses.reshape([num_states, state_dim])
        return state_guesses

    def opt_vars_to_states_and_lagrange(
        self, opt_vars, pos_init, pos_end_targ, num_states
    ):
        if len(pos_init.shape) == 1:
            pos_dim = pos_init.shape[0]
            state_dim = 2 * pos_dim
        state_guesses = self.opt_vars_to_states(
            opt_vars, pos_init, pos_end_targ, num_states
        )

        lagrange_multipliers = opt_vars[num_states * state_dim - 2 * pos_dim :]
        return state_guesses, lagrange_multipliers

    def lagrange_objective(self, opt_vars, pos_init, pos_end_targ, times):
        (
            state_guesses,
            lagrange_multipliers,
        ) = self.opt_vars_to_states_and_lagrange(
            opt_vars, pos_init, pos_end_targ, num_states=times.shape[0]
        )

        eq_constraints = self.collocation_defects(opt_vars)
        lagrange_term = jnp.sum(lagrange_multipliers * eq_constraints)
        objective = self.sum_of_squares_objective(
            opt_vars, pos_init, pos_end_targ, times
        )
        lagrange_objective = objective - lagrange_term
        return lagrange_objective

    def solve_trajectory_lagrange_jax(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
        step_size=0.1,
        states_tol=0.2,
    ):
        self.state_guesses = state_guesses
        self.pos_init = pos_init
        self.pos_end_targ = pos_end_targ
        # hack as times needed in collocation_constraints_fn
        self.times = times  # TODO delete this!!

        if len(pos_init.shape) == 1:
            pos_dim = pos_init.shape[0]
        state_dim = state_guesses.shape[1]
        num_states = state_guesses.shape[0]

        state_guesses_vars = state_guesses_to_opt_vars(state_guesses)

        # Initialise lagrange mutlipliers for collocation defects
        num_defects = num_states - 1
        lagrange_multipliers = jnp.zeros([num_defects * state_dim])
        opt_vars = jnp.concatenate(
            [state_guesses_vars, lagrange_multipliers], axis=0
        )
        opt_vars_vars = objax.StateVar(opt_vars)

        # Initialise lagrange objective fn with collocation defect constraints
        objective_args = (pos_init, pos_end_targ, times)
        jitted_fn_vars = objax.VarCollection({"opt_vars": opt_vars_vars})
        # jitted_lagrange_objective = objax.Jit(
        #     self.lagrange_objective, jitted_fn_vars
        # )


        def lagrange_objective(l):
            return self.lagrange_objective(l, *objective_args)
            # return jitted_lagrange_objective(l, *objective_args)

        L = jax.jacfwd(lagrange_objective)
        gradL = jax.jacfwd(L)

        def optimiser_step(l):
            return l - step_size * jnp.linalg.inv(gradL(l)) @ L(l)

        jitted_optimiser_step = objax.Jit(optimiser_step, jitted_fn_vars)

        states = self.opt_vars_to_states(
            opt_vars, pos_init, pos_end_targ, num_states
        )
        print("tol", states_tol * step_size)
        for epoch in range(self.maxiter):
            print("Epoch: ", epoch)
            opt_vars = jitted_optimiser_step(opt_vars)
            states_next = self.opt_vars_to_states(
                opt_vars, pos_init, pos_end_targ, num_states
            )
            states_diff = jnp.sum(jnp.absolute(states_next - states))
            states = states_next
            print(states_diff)
            if states_diff < states_tol * step_size:
                break

        self.optimisation_result = opt_vars
        print("Optimisation Result")
        print(self.optimisation_result)
        opt_vars = self.optimisation_result

        (
            self.optimised_trajectory,
            lagrange_multipliers,
        ) = self.opt_vars_to_states_and_lagrange(
            opt_vars, pos_init, pos_end_targ, num_states
        )

        print("Optimised Trajectory")
        print(self.optimised_trajectory)
        return self.optimised_trajectory

    def solve_trajectory_lagrange(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
    ):
        self.state_guesses = state_guesses
        self.pos_init = pos_init
        self.pos_end_targ = pos_end_targ
        # hack as times needed in collocation_constraints_fn
        self.times = times  # TODO delete this!!

        method = "SLSQP"
        # method = "L-BFGS-B"

        if len(pos_init.shape) == 1:
            pos_dim = pos_init.shape[0]
            # state_dim = 2 * pos_dim
        state_dim = state_guesses.shape[1]
        num_states = state_guesses.shape[0]

        state_guesses_vars = state_guesses_to_opt_vars(state_guesses)

        # Initialise lagrange mutlipliers for collocation defects
        num_defects = num_states - 1
        # lagrange_multipliers = 0.01 * jnp.ones([num_defects * state_dim])
        # lagrange_multipliers = 1./5000.0*jnp.ones([num_defects * state_dim])
        # lagrange_multipliers = 1.0 / 5000.0 * jnp.ones([num_defects * pos_dim])
        lagrange_multipliers = jnp.zeros([num_defects * state_dim])
        print("lagrange_multipliers")
        print(lagrange_multipliers.shape)
        opt_vars = jnp.concatenate(
            [state_guesses_vars, lagrange_multipliers], axis=0
        )
        print("opt_vars")
        print(opt_vars.shape)
        opt_vars_vars = objax.StateVar(opt_vars)

        # Initialise lagrange objective fn with collocation defect constraints
        objective_args = (pos_init, pos_end_targ, times)
        jitted_fn_vars = objax.VarCollection({"opt_vars": opt_vars_vars})
        jitted_lagrange_objective = objax.Jit(
            self.lagrange_objective, jitted_fn_vars
        )
        # jitted_objective = objax.Jit(self.objective_fn, jitted_fn_vars)
        # lag = self.lagrange_objective(opt_vars, pos_init, pos_end_targ, times)
        # print('after lag fun call')
        # print(lag.shape)
        def jac_fn(opt_vars, pos_init, pos_end_targ, times):
            jac_fn_ = jax.jacfwd(self.lagrange_objective)
            return jac_fn_(opt_vars, pos_init, pos_end_targ, times)

        jitted_jac_fn = objax.Jit(jac_fn, jitted_fn_vars)

        self.optimisation_result = sp.optimize.minimize(
            # self.lagrange_objective,
            jitted_lagrange_objective,
            opt_vars,
            # jac=jitted_jac_fn,
            # jac=jac_fn,
            method=method,
            options={"disp": True, "maxiter": self.maxiter},
            args=objective_args,
        )
        print("Optimisation Result")
        print(self.optimisation_result)
        opt_vars = self.optimisation_result.x

        (
            self.optimised_trajectory,
            lagrange_multipliers,
        ) = self.opt_vars_to_states_and_lagrange(
            opt_vars, pos_init, pos_end_targ, num_states
        )
        print("Optimised Trajectory")
        print(self.optimised_trajectory)
        return self.optimised_trajectory

    def solve_trajectory(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
        lb_defect=-0.05,
        ub_defect=0.05,
        bounds: Bounds = None,
    ):
        self.state_guesses = state_guesses
        self.pos_init = pos_init
        self.pos_end_targ = pos_end_targ
        # hack as times needed in collocation_constraints_fn
        self.times = times  # TODO delete this!!

        method = "SLSQP"

        # bound the start and end (x,y) positions in the state vector
        # if bounds is None:
        #     # bounds = start_end_pos_bounds(
        #     #     state_guesses, pos_init, pos_end_targ
        #     # )
        #     bounds = start_end_pos_bounds_lagrange(
        #         state_guesses,
        #         pos_init,
        #         pos_end_targ,
        #         pos_init_idx=0,
        #         pos_end_idx=-1,
        #         tol=0.02,
        #     )

        states_shape = state_guesses.shape
        state_guesses_vars = state_guesses_to_opt_vars(state_guesses)
        # state_guesses = state_guesses.reshape(-1)
        # state_guesses_StateVar = objax.StateVar(state_guesses_vars)

        # Initialise collocation defects as constraints
        jitted_fn_vars = objax.VarCollection(
            {"state_guesses": objax.StateVar(state_guesses_vars)}
        )
        jitted_collocation_defects = objax.Jit(
            self.collocation_defects, jitted_fn_vars
        )
        jitted_collocation_constraints = NonlinearConstraint(
            jitted_collocation_defects,
            lb_defect,
            ub_defect,
        )
        collocation_constraints = NonlinearConstraint(
            self.collocation_defects,
            lb_defect,
            ub_defect,
        )

        # Initialise objective function
        objective_args = (pos_init, pos_end_targ, times)
        jitted_dummy_objective_fn = objax.Jit(
            self.dummy_objective_fn, jitted_fn_vars
        )
        jitted_sum_of_squares_objective = objax.Jit(
            self.sum_of_squares_objective, jitted_fn_vars
        )
        jitted_objective_fn = objax.Jit(self.objective_fn, jitted_fn_vars)

        self.optimisation_result = sp.optimize.minimize(
            # jitted_objective_fn,
            jitted_sum_of_squares_objective,
            # jitted_dummy_objective_fn,
            # self.dummy_objective_fn,
            # state_guesses,
            state_guesses_vars,
            method=method,
            # bounds=bounds,
            constraints=jitted_collocation_constraints,
            # constraints=collocation_constraints,
            options={"disp": True, "maxiter": self.maxiter},
            args=objective_args,
        )
        print("Optimisation Result")
        print(self.optimisation_result)
        opt_vars = self.optimisation_result.x
        # opt_vars= self.optimisation_result.x.reshape(
        #     states_shape
        # )

        self.optimised_trajectory = self.opt_vars_to_states(
            opt_vars, pos_init, pos_end_targ, states_shape[0]
        )
        # state_opt = state_opt.reshape(states_shape)
        print("Optimised Trajectory")
        print(self.optimised_trajectory)
        return self.optimised_trajectory
