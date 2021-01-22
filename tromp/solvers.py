import jax
import jax.numpy as np
from jax.experimental.ode import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import root

from tromp.ode import geodesic_ode


def integrate_ode_fn(
    ode_fn, state_init, times, t_init=0.0, t_end=1.0, int_method="RK45"
):
    integrator = solve_ivp(
        fun=ode_fn,
        t_span=(t_init, t_end),
        y0=state_init,
        t_eval=times,
        # r_tol=1.,
        # a_tol=1.,
        method=int_method,
    )
    return integrator.t, integrator.y


def error_geodesic_vel(
    vel_init,
    pos_init,
    pos_end_targ,
    metric_fn,
    metric_fn_args,
    times,
    t_init=0.0,
    t_end=1.0,
    int_method="RK45",
):
    def ode_fn(t, state):
        return geodesic_ode(t, state, metric_fn, metric_fn_args)

    input_dim = pos_init.shape[0]
    print("Shooting with initial velocity: ", vel_init)
    state_init = np.concatenate([pos_init, vel_init])

    ts, states = integrate_ode_fn(
        ode_fn,
        state_init,
        times,
        t_init=t_init,
        t_end=t_end,
        int_method=int_method,
    )
    pos_end_integrated = np.array(states)[0:input_dim, -1]

    error = pos_end_targ - pos_end_integrated
    print("Target pos error: ", error)
    return error


def shooting_geodesic_solver(
    pos_init,
    pos_end_targ,
    vel_init_guess,
    metric_fn,
    metric_fn_kwargs,
    times,
    t_init=0.0,
    t_end=1.0,
    int_method="RK45",
    root_tol=0.05,
    maxfev=1000,
):
    def ode_fn(t, state):
        return geodesic_ode(t, state, metric_fn, metric_fn_kwargs)

    error_geodesic_vel_args = (
        pos_init,
        pos_end_targ,
        metric_fn,
        metric_fn_kwargs,
        times,
        t_init,
        t_end,
        int_method,
    )
    root_method = "krylov"
    root_method = "lm"
    opt_result = root(
        error_geodesic_vel,
        vel_init_guess,
        args=error_geodesic_vel_args,
        # jac=loss_jac,
        method=root_method,
        options={"maxfev": maxfev, "disp": True},
        tol=root_tol,
    )
    print("Root finding terminated...")
    print(opt_result)
    opt_vel_init = opt_result.x

    state_init = np.concatenate([pos_init, opt_vel_init])
    _, geodesic_traj = integrate_ode_fn(
        ode_fn,
        state_init,
        times,
        t_init=t_init,
        t_end=t_end,
        int_method=int_method,
    )
    return opt_vel_init, geodesic_traj


def integrate_guesses_scipy(ode_fn, state_guesses, times, int_method="RK45"):
    t_init = times[0]
    t_end = times[-1]
    times = np.linspace(t_init, t_end, 1000)
    integrator = solve_ivp(
        fun=ode_fn,
        t_span=(t_init, t_end),
        y0=state_guesses,
        t_eval=times,
        method=int_method,
    )
    end_state_integrated = np.array(integrator.y)[:, -1]
    return end_state_integrated


def integrate_multiple_guesses(state_guesses, times, ode_args):
    def ode_fn_scipy(t, state):
        # print('inside ode_fn')
        return geodesic_ode(t, state, *ode_args)

    num_grid_points = state_guesses.shape[0]
    states_integrated = np.empty(
        [state_guesses.shape[0], state_guesses.shape[1]]
    )
    for i in range(num_grid_points - 1):
        state_integrated = integrate_guesses_scipy(
            ode_fn_scipy, state_guesses[i, :], times[i, :]
        )
        states_integrated = jax.ops.index_update(
            states_integrated, jax.ops.index[i + 1, :], state_integrated
        )
        # print('grid point ', str(i))
        # print(states_integrated)
    return states_integrated


# @jax.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def shooting_error(
    state_guesses, pos_start, pos_end, metric_fn, metric_fn_kwargs, times
):
    def ode_fn(state, t):
        print("inside ode_fn")
        return geodesic_ode(t, state, metric_fn, metric_fn_kwargs)

    def integrate_guesses(ode_fn, state_init, times):
        states_integrated = odeint(ode_fn, state_init, times)
        end_state_integrated = np.array(states_integrated)[-1, :]
        return end_state_integrated

    input_dim = pos_start.shape[0]
    state_guesses = state_guesses.reshape([-1, 2 * input_dim])
    state_guesses = jax.ops.index_update(
        state_guesses, jax.ops.index[0, :input_dim], pos_start
    )
    state_guesses = jax.ops.index_update(
        state_guesses, jax.ops.index[-1, :input_dim], pos_end
    )
    # print('state guesses after update')
    # print(state_guesses)
    ode_args = [metric_fn, metric_fn_kwargs]
    states_integrated = integrate_multiple_guesses(
        state_guesses, times, ode_args
    )
    # print('states integrated')
    # print(states_integrated)
    states_integrated = jax.ops.index_update(
        states_integrated, jax.ops.index[0, :], state_guesses[0, :]
    )
    # print(states_integrated)
    # states_integrated = integrate_guesses(ode_fn, state_guesses[0, :], times)
    # states_integrated = jax.vmap(integrate_guesses,
    #                              (None, 0, 0))(ode_fn, state_guesses, times)
    # pos_guesses = state_guesses[:, 0:input_dim]
    # pos_guesses = pos_guesses[1:, :]
    # pos_integrated = states_integrated[:, 0:input_dim]

    error = np.subtract(states_integrated, state_guesses)
    # print('error')
    # print(error.shape)
    # error = jax.ops.index_update(error, jax.ops.index[-1, :],
    #                              np.zeros([2 * input_dim]))
    # error = jax.ops.index_update(error, jax.ops.index[-1, input_dim:],
    #                              np.zeros([input_dim]))

    print("Target pos error: ", error)
    error = error.reshape([-1])
    return error


# @jax.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def multiple_shooting(state_guesses, metric_fn, metric_fn_kwargs, times):
    input_dim = int(state_guesses.shape[-1] / 2)
    # pos_guesses = state_guesses[1:-1, 0:input_dim]
    # vel_guesses = state_guesses[:, input_dim:]
    pos_start = state_guesses[0, 0:input_dim]
    pos_end = state_guesses[-1, 0:input_dim]
    shooting_error_args = (
        pos_start,
        pos_end,
        metric_fn,
        metric_fn_kwargs,
        times,
    )

    # root_method = 'krylov'
    root_method = "hybr"
    maxfev = 1000
    root_tol = 0.0005
    print("inside multiple shooting")
    print(state_guesses.shape)
    state_guesses = state_guesses.flatten()
    print(state_guesses)
    print(state_guesses.shape)
    opt_result = root(
        shooting_error,
        state_guesses,
        args=shooting_error_args,
        method=root_method,
        options={"maxfev": maxfev, "disp": True},
        tol=root_tol,
    )
    print("Root finding terminated...")
    print(opt_result)
    # opt_vel_init = opt_result.x
    state_opt = opt_result.x
    print(state_opt.shape)
    state_opt = state_opt.reshape([-1, 2 * input_dim])

    state_opt = jax.ops.index_update(
        state_opt, jax.ops.index[-1, :input_dim], pos_end
    )
    print(state_opt.shape)
    print(state_opt)

    # ode_args = [metric_fn, metric_fn_kwargs]
    # geodesic_traj = integrate_multiple_guesses(state_opt, times, ode_args)
    # print('traj')
    # print(geodesic_traj.shape)
    return state_opt
    # return state_opt, geodesic_traj
