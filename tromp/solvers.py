import abc
import time
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import objax
import scipy as sp
from bunch import Bunch
from jax.config import config
from scipy.optimize import Bounds, NonlinearConstraint

from tromp.constraints import hermite_simpson_collocation_constraints_fn

# from tromp.helpers import init_start_end_pos_scipy_bounds
from tromp.metric_tensors import RiemannianMetricTensor

# from tromp.ode import geodesic_ode
from tromp.ode import ODE, GeodesicODE

config.update("jax_enable_x64", True)


def constant_objective_fn(
    state_guesses,
    pos_init,
    pos_end_targ,
    times,
):
    return 1.0


def sum_of_squares_objective(
    opt_vars,
    pos_init,
    pos_end_targ,
    times,
):
    if len(pos_init.shape) == 1:
        pos_dim = pos_init.shape[0]
    num_states = times.shape[0]
    state_guesses = opt_vars_to_states(opt_vars, pos_init, pos_end_targ, num_states)
    pos_guesses = state_guesses[:, :pos_dim]
    sum_of_squares = jnp.sum(pos_guesses ** 2)
    # sum_of_squares = jnp.sum(state_guesses ** 2)
    return sum_of_squares
    # return sum_of_squares * 1000


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
    state_guesses = jnp.concatenate([state_guesses_before, state_guesses_end], axis=0)
    print("state guesses removed end pos")
    print(state_guesses.shape)
    return state_guesses


def opt_vars_to_states(opt_vars, pos_init, pos_end_targ, num_states):
    if len(pos_init.shape) == 1:
        pos_dim = pos_init.shape[0]
        state_dim = 2 * pos_dim
    state_guesses = opt_vars[: num_states * state_dim - 2 * pos_dim]
    # Add start pos
    state_guesses = jnp.concatenate([pos_init, state_guesses], axis=0)

    # Split state_guesses and insert end pos
    state_guesses_before = state_guesses[:-pos_dim]
    vel_end = state_guesses[-pos_dim:]
    state_guesses = jnp.concatenate([state_guesses_before, pos_end_targ], axis=0)
    state_guesses = jnp.concatenate([state_guesses, vel_end], axis=0)

    state_guesses = state_guesses.reshape([num_states, state_dim])
    return state_guesses


def opt_vars_to_states_and_lagrange(opt_vars, pos_init, pos_end_targ, num_states):
    if len(pos_init.shape) == 1:
        pos_dim = pos_init.shape[0]
        state_dim = 2 * pos_dim
    state_guesses = opt_vars_to_states(opt_vars, pos_init, pos_end_targ, num_states)

    lagrange_multipliers = opt_vars[num_states * state_dim - 2 * pos_dim :]
    return state_guesses, lagrange_multipliers


# class BaseSolver(objax.Module, abc.ABC):
#     def __init__(self, ode: ODE):
#         self.ode = ode
#         # self.times = times

    # @abc.abstractmethod
    # def objective_fn(self, state):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def solve_trajectory(
    #     self,
    #     state_guesses,
    #     pos_init,
    #     pos_end_targ,
    #     times,
    # ):
    #     raise NotImplementedError


class GeodesicSolver(objax.Module, abc.ABC):
    def __init__(self, ode: GeodesicODE):
        self.ode = ode
        # self.times = times

    def save(self, filename=None):
        if filename is None:
            filename = "./GeodesicSolver.pickle"
        with open(filename, "wb") as path:
            pickle.dump(self, path, -1)

    def constant_objective_fn(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
    ):
        return 1.0

    def metric_objective_fn(
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
        state_guesses = opt_vars_to_states(opt_vars, pos_init, pos_end_targ, num_states)
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
        state_guesses = opt_vars_to_states(opt_vars, pos_init, pos_end_targ, num_states)
        pos_guesses = state_guesses[:, :pos_dim]
        sum_of_squares = jnp.sum(pos_guesses ** 2)
        # sum_of_squares = jnp.sum(state_guesses ** 2)
        return sum_of_squares * 1000

    # @abc.abstractmethod
    # def objective_fn(self, state):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def solve_trajectory(
    #     self,
    #     state_guesses,
    #     pos_init,
    #     pos_end_targ,
    #     times,
    # ):
    #     raise NotImplementedError


class CollocationGeodesicSolver(GeodesicSolver):
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

    def objective_fn(self, state_guesses, pos_init, pos_end_targ, times):
        return sum_of_squares_objective(state_guesses, pos_init, pos_end_targ, times)

    def lagrange_objective(self, opt_vars, pos_init, pos_end_targ, times):
        (state_guesses, lagrange_multipliers,) = opt_vars_to_states_and_lagrange(
            opt_vars, pos_init, pos_end_targ, num_states=times.shape[0]
        )

        eq_constraints = hermite_simpson_collocation_constraints_fn(
            state_at_knots=state_guesses, times=times, ode_fn=self.ode.ode_fn
        )
        print("eq")
        print(eq_constraints.shape)
        lagrange_term = jnp.sum(lagrange_multipliers * eq_constraints)

        # state_guesses_vars = state_guesses_to_opt_vars(state_guesses)
        # print(state_guesses_vars.shape)
        # def eq_constraints_fn(state_guesses_vars):
        #     state_at_knots = opt_vars_to_states(
        #         state_guesses_vars, pos_init, pos_end_targ, num_states=times.shape[0]
        #     )
        #     # state_at_knots = state_at_knots.reshape(state_guesses_shape)
        #     return hermite_simpson_collocation_constraints_fn(
        #         state_at_knots, times=times, ode_fn=self.ode.ode_fn
        #     )

        # jacobian_eq_constraints_fn = jax.jacfwd(eq_constraints_fn)
        # jacobian_eq_constraints = jacobian_eq_constraints_fn(state_guesses_vars)
        # print("jac consts")
        # print(jacobian_eq_constraints.shape)
        # print(lagrange_multipliers.shape)
        # print(lagrange_multipliers.shape)
        # # lagrange_multipliers = lagrange_multipliers.reshape([-1, 1])
        # # lagrange_term = jnp.sum(lagrange_multipliers * jacobian_eq_constraints, axis=0)
        # lagrange_term = jnp.sum(lagrange_multipliers * jacobian_eq_constraints)
        # print("lag term")
        # print(lagrange_term.shape)

        # objective = self.sum_of_squares_objective(
        #     opt_vars, pos_init, pos_end_targ, times
        # )
        # lagrange_objective = objective - lagrange_term
        lagrange_objective = -lagrange_term
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
        """
        Optimiser uses Newton's method
        """
        self.state_guesses = state_guesses

        if len(pos_init.shape) == 1:
            pos_dim = pos_init.shape[0]
        state_dim = state_guesses.shape[1]
        num_states = state_guesses.shape[0]

        state_guesses_vars = state_guesses_to_opt_vars(state_guesses)

        # Initialise lagrange mutlipliers for collocation defects
        num_defects = num_states - 1
        lagrange_multipliers = jnp.ones([num_defects * state_dim])
        opt_vars = jnp.concatenate([state_guesses_vars, lagrange_multipliers], axis=0)
        # opt_vars_vars = objax.StateVar(opt_vars)
        opt_vars = objax.TrainVar(opt_vars)
        opt_vars_ref = objax.TrainRef(opt_vars)

        # Initialise lagrange objective fn with collocation defect constraints
        objective_args = (pos_init, pos_end_targ, times)
        jitted_fn_vars = objax.VarCollection({"opt_vars": opt_vars})
        # jitted_lagrange_objective = objax.Jit(
        #     self.lagrange_objective, jitted_fn_vars
        # )

        def lagrange_objective(l):
            return self.lagrange_objective(l, *objective_args)
            # return jitted_lagrange_objective(l, *objective_args)

        jacobian_lagrange_objective = jax.jacfwd(lagrange_objective)
        hessian_lagrange_objective = jax.jacfwd(jacobian_lagrange_objective)

        def optimiser_step(opt_vars):
            jacobian_loss = jacobian_lagrange_objective(opt_vars)
            loss = lagrange_objective(opt_vars)
            updated_opt_vars = (
                opt_vars
                - step_size
                * jnp.linalg.inv(hessian_lagrange_objective(opt_vars))
                @ jacobian_loss
            )
            return updated_opt_vars, loss

        jitted_optimiser_step = objax.Jit(optimiser_step, jitted_fn_vars)

        states = opt_vars_to_states(
            opt_vars_ref.value, pos_init, pos_end_targ, num_states
        )

        print("tol", states_tol / step_size)
        self.loss_at_steps = []
        self.opt_vars_at_steps = []
        for step in range(self.maxiter):
            t = time.time()
            # opt_vars, loss = jitted_optimiser_step(opt_vars)
            # opt_vars, loss = jitted_optimiser_step(opt_vars_ref)
            opt_vars_ref.value, loss = jitted_optimiser_step(opt_vars_ref.value)
            self.loss_at_steps.append(loss)
            self.opt_vars_at_steps.append(opt_vars_ref.value)
            states_next = opt_vars_to_states(
                opt_vars_ref.value, pos_init, pos_end_targ, num_states
            )
            states_diff = jnp.sum(jnp.absolute(states_next - states))
            states = states_next
            duration = time.time() - t
            print("Loss @ step {} is {}".format(step, loss))
            # print("Step {}".format(step))
            print("Duration: ", duration)
            # print(states_diff)
            # if states_diff < states_tol / step_size:
            #     break
        plt.plot(jnp.arange(len(self.loss_at_steps)), self.loss_at_steps)
        plt.show()

        min_loss_idx = np.nanargmin(np.array(self.loss_at_steps))
        self.optimisation_result = self.opt_vars_at_steps[min_loss_idx]
        print("Optimisation Result")
        print(self.optimisation_result)

        (
            self.optimised_trajectory,
            lagrange_multipliers,
        ) = opt_vars_to_states_and_lagrange(
            self.optimisation_result, pos_init, pos_end_targ, num_states
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

        def lagrange_objective(l):

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


class ScipyCollocationGeodesicSolver(GeodesicSolver):
    def __init__(
        self,
        ode,
        covariance_weight: jnp.float64 = 1.0,
        maxiter: int = 100,
    ):
        super().__init__(ode)
        self.covariance_weight = covariance_weight
        self.maxiter = maxiter

    def objective_fn(self, state_guesses, pos_init, pos_end_targ, times):
        return sum_of_squares_objective(state_guesses, pos_init, pos_end_targ, times)

    def solve_trajectory(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
        lb_defect=-0.05,
        ub_defect=0.05,
        method="SLSQP",
        jit=True,
        disp=True,
    ):
        # store initial state trajectory for comaprison/plotting etc
        self.state_guesses = state_guesses

        def collocation_constraints_fn(opt_vars):
            state_guesses = opt_vars_to_states(
                opt_vars, pos_init, pos_end_targ, num_states=times.shape[0]
            )
            return hermite_simpson_collocation_constraints_fn(
                state_at_knots=state_guesses,
                times=times,
                ode_fn=self.ode.ode_fn,
            )

        (
            self.optimised_trajectory,
            self.optimisation_result,
        ) = self.solve_trajectory_scipy_minimize(
            state_guesses,
            pos_init,
            pos_end_targ,
            times,
            constraints_fn=collocation_constraints_fn,
            objective_fn=self.objective_fn,
            lb_defect=lb_defect,
            ub_defect=ub_defect,
            method=method,
            jit=jit,
            disp=disp,
        )
        return self.optimised_trajectory

    def solve_trajectory_scipy_minimize(
        self,
        state_guesses,
        pos_init,
        pos_end_targ,
        times,
        constraints_fn=None,
        objective_fn=None,
        lb_defect=0.01,
        ub_defect=0.01,
        method="SLSQP",
        jit=True,
        disp=True,
    ):
        print("inside scipy solve traj")
        print(state_guesses.shape)
        states_shape = state_guesses.shape
        state_guesses_vars = state_guesses_to_opt_vars(state_guesses)
        print(state_guesses_vars.shape)

        objective_args = (pos_init, pos_end_targ, times)
        if jit:
            jitted_fn_vars = objax.VarCollection(
                {"state_guesses": objax.StateVar(state_guesses_vars)}
            )

            # Initialise objective function
            objective_fn = objax.Jit(objective_fn, jitted_fn_vars)

        # If constraints then initialise (and maybe jit)
        if constraints_fn is not None:
            if jit:
                constraints_fn = objax.Jit(constraints_fn, jitted_fn_vars)
            constraints = NonlinearConstraint(
                constraints_fn,
                lb_defect,
                ub_defect,
            )

        optimisation_result = sp.optimize.minimize(
            objective_fn,
            state_guesses_vars,
            method=method,
            constraints=constraints,
            options={"disp": disp, "maxiter": self.maxiter},
            args=objective_args,
        )
        print("Optimisation Result:")
        print(optimisation_result)

        optimised_trajectory = opt_vars_to_states(
            optimisation_result.x, pos_init, pos_end_targ, states_shape[0]
        )
        print("Optimised Trajectory:")
        print(optimised_trajectory)
        return optimised_trajectory, optimisation_result

        # bound the start and end (x,y) positions in the state vector
        # if bounds is None:
        #     # bounds = start_end_pos_bounds(
        #     #     state_guesses, pos_init, pos_end_targ
        #     # )
        #     bounds = init_start_end_pos_scipy_bounds(
        #         state_guesses,
        #         pos_init,
        #         pos_end_targ,
        #         pos_init_idx=0,
        #         pos_end_idx=-1,
        #         tol=0.02,
        #     )
