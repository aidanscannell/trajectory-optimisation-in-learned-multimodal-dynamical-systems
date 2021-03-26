import jax
from jax.config import config


config.update("jax_enable_x64", True)


def states_at_mid_points_hermite_interpolation(
    state_at_knots, state_prime_at_knots, delta_times
):
    """Approximate states at mid points using Hermite interpolation

    The Hermite interpolant is given by,
        x_{k+1/2} = 0.5 (x_{k} + x_{k+1}) + h_{k}/8 (f_{k} - f_{k+1})

    :param state_at_knots: states at knot points [num_states, state_dim]
    :param state_prime_at_knots: states derivatives at knot points [num_states, state_dim]
    :param delta_times: delta time between knot points [num_states-1, 1] e.g. t_{k+1} - t_{k}
    :returns: interpolated states at mid points [num_states-1, state_dim]

    """
    state_at_knots_before = state_at_knots[0:-1, :]
    state_at_knots_after = state_at_knots[1:, :]
    state_prime_at_knots_before = state_prime_at_knots[0:-1, :]
    state_prime_at_knots_after = state_prime_at_knots[1:, :]
    state_at_mid_points = 0.5 * (
        state_at_knots_before + state_at_knots_after
    ) + delta_times / 8 * (state_prime_at_knots_before - state_prime_at_knots_after)
    # state_at_mid_points = 0.5 * (state_at_knots_before + state_at_knots_after) + dt / 8 * (
    #     state_prime_at_knots_before - state_prime_at_knots_after
    # )
    return state_at_mid_points


def hermite_simpson_collocation_constraints_fn(state_at_knots, times, ode_fn):
    """Return Hermite-Simpson collocation constraints

    for state_prime = ode_fn(state_guesses)

    The continuos dynamics are transcribed to a set of collocation equations when

    Simpson quadrature approximates the integrand of the integral as a piecewise quadratic function

    The state trajectory is a cubic Hermite spline, which has a continuous first derivative.

    The collocation constraints returned are in compressed form.

    The Simpson collocation defect is given by,
        0 = x_{i+1} - \hat{x}_{i}
          = x_{i+1} - (x_{i} + h_{i} f_{i})
          = x_{i+1} - x_{i} - h_{i+1}/6 (f_{i} + 4*\hat{f}_{i} + f_{i+1})
    where h_{i} f_{i} is the interpolant.
    The Hermite interpolant is given by,
        0.5 (x_{i} + x_{i+1}) + h_{i+1}/8 (f_{j} - f_{j+1})

    At each knot point the state and its derivative should equal those from the system dynamics (ode_fn)

    :param state_guesses: states at knot points [num_states, state_dim]
    :param ode_fn:
    :returns: collocation defects [num_states-1, state_dim]
    """
    # num_timesteps = times.shape[0]
    num_states = times.shape[0]
    print("inside hermite simpson defects")
    print(state_at_knots.shape)
    print(times.shape)

    # dt = times[-1] - times[0]
    # dt = times[1] - times[0]
    times_before = times[0:-1]
    times_after = times[1:]
    delta_times = times_after - times_before
    # times_at_mid_points = jnp.linspace(
    #     times[0] + delta_times[0] / 2,
    #     times[-1] - delta_times[0] / 2,
    #     num_states - 1,
    # )
    # dt = times_after - times_before
    print("delta_times")
    print(delta_times.shape)
    delta_times = delta_times.reshape(-1, 1)
    print(delta_times.shape)

    def ode_fn_single_state(state):
        return ode_fn(times, state)

    # TODO time should be used inside ODE
    # def ode_fn_single(time, state):
    #     return ode_fn(state)

    # State derivative according to the true continuous dynamics
    state_prime_at_knots = jax.vmap(ode_fn_single_state)(state_at_knots)
    # state_prime_at_knots = ode_fn(times, state_at_knots)

    # Approx states at mid points using Hermite interpolation
    state_at_knots_before = state_at_knots[0:-1, :]
    state_at_knots_after = state_at_knots[1:, :]
    state_prime_at_knots_before = state_prime_at_knots[0:-1, :]
    state_prime_at_knots_after = state_prime_at_knots[1:, :]
    state_at_mid_points = states_at_mid_points_hermite_interpolation(
        # state_at_knots, state_prime_at_knots, times_mid_points=times_mid_points
        state_at_knots,
        state_prime_at_knots,
        delta_times=delta_times,
    )

    print("state at mid ")
    print(state_at_mid_points.shape)
    # Calc state derivatives at mid points
    state_prime_at_mid_points = jax.vmap(ode_fn_single_state)(state_at_mid_points)
    # state_prime_at_mid_points = ode_fn(
    #     times_at_mid_points, state_at_mid_points
    # )

    # defects = (state_at_knots_before - state_at_knots_after) + dt / 6 * (
    # defects = (state_at_knots_after - state_at_knots_before) - dt / 6 * (
    defects = (state_at_knots_after - state_at_knots_before) - delta_times / 6 * (
        state_prime_at_knots_before
        + 4 * state_prime_at_mid_points
        + state_prime_at_knots_after
    )
    print("Collcation Defects")
    print(defects)
    return defects.flatten()


# def hermite_simpson_trajectory():


# def init_scipy_collocation_constraints(jit=True):
#     # Initialise collocation defects as constraints
#     if jit:
#         jitted_collocation_defects = objax.Jit(
#             collocation_defects, jitted_fn_vars
#         )
#         jitted_collocation_constraints = NonlinearConstraint(
#             jitted_collocation_defects,
#             lb_defect,
#             ub_defect,
#         )
#     else:
#         collocation_constraints = NonlinearConstraint(
#             collocation_defects,
#             lb_defect,
#             ub_defect,
#         )


def state_prime_at_midpoints_ode(state_at_knots, times, ode_fn):
    print("here")
    print(state_at_knots.shape)
    times_before = times[0:-1]
    times_after = times[1:]
    delta_times = times_after - times_before
    delta_times = delta_times.reshape(-1, 1)

    def ode_fn_single_state(state):
        return ode_fn(times, state)

    # State derivative according to the true continuous dynamics
    state_prime_at_knots = jax.vmap(ode_fn_single_state)(state_at_knots)
    # state_prime_at_knots = ode_fn(times, state_at_knots)

    # Approx states at mid points using Hermite interpolation
    state_at_mid_points = states_at_mid_points_hermite_interpolation(
        state_at_knots,
        state_prime_at_knots,
        delta_times=delta_times,
    )

    # Calculate state derivatives at mid points
    state_prime_at_mid_points = jax.vmap(ode_fn_single_state)(state_at_mid_points)
    return state_prime_at_mid_points


def state_prime_at_midpoints_hermite_interpolation(state_at_knots, times, ode_fn):
    times_before = times[0:-1]
    times_after = times[1:]
    delta_times = times_after - times_before
    delta_times = delta_times.reshape(-1, 1)

    def ode_fn_single_state(state):
        return ode_fn(times, state)

    # State derivative according to the true continuous dynamics
    state_prime_at_knots = jax.vmap(ode_fn_single_state)(state_at_knots)
    # state_prime_at_knots = ode_fn(times, state_at_knots)

    # Approx states at mid points using Hermite interpolation
    state_at_knots_before = state_at_knots[0:-1, :]
    state_at_knots_after = state_at_knots[1:, :]
    state_prime_at_knots_before = state_prime_at_knots[0:-1, :]
    state_prime_at_knots_after = state_prime_at_knots[1:, :]

    # Calculate interpolated state prime
    interpolated_state_prime_at_mid_points = -3 / 2 * delta_times * (
        state_at_knots_before - state_at_knots_after
    ) - 0.25 * (state_prime_at_knots_before + state_prime_at_knots_after)
    return -interpolated_state_prime_at_mid_points
