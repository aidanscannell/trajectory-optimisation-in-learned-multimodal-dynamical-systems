from jax import numpy as np
from jax import jit, partial
from scipy.integrate import solve_ivp
from scipy.optimize import root

from ProbGeo.ode import geodesic_ode


def integrate_ode_fn(ode_fn,
                     state_init,
                     times,
                     t_init=0.,
                     t_end=1.,
                     int_method='RK45'):
    integrator = solve_ivp(
        fun=ode_fn,
        t_span=(t_init, t_end),
        y0=state_init,
        t_eval=times,
        # max_step=max_step,
        method=int_method)
    # print(integrator)
    return integrator.t, integrator.y


# @partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def residuals_geodesic_vel(vel_init,
                           pos_init,
                           pos_end_targ,
                           metric_fn,
                           metric_fn_args,
                           times,
                           t_init=0.,
                           t_end=1.,
                           int_method='RK45'):
    # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
    def ode_fn(t, state):
        return geodesic_ode(t, state, metric_fn, metric_fn_args)

    # TODO make input_dim dynamic
    input_dim = 2
    print('Trying vel[t=0]: ', vel_init)
    state_init = np.concatenate([pos_init, vel_init])

    ts, states = integrate_ode_fn(ode_fn,
                                  state_init,
                                  times,
                                  t_init=t_init,
                                  t_end=t_end,
                                  int_method=int_method)
    pos_end_integrated = np.array(states)[0:input_dim, -1]
    print('end integrated:', pos_end_integrated)
    print('end targer:', pos_end_targ)

    error = pos_end_targ - pos_end_integrated
    print('Residual [pos_target(t_end)-pos_integrated(t_end)]: ', error)
    return error


def shooting_geodesic_solver(pos_init,
                             pos_end_targ,
                             vel_init_guess,
                             metric_fn,
                             metric_fn_args,
                             times,
                             t_init=0.,
                             t_end=1.,
                             int_method='RK45',
                             root_tol=0.05,
                             maxfev=1000):
    def ode_fn(t, state):
        return geodesic_ode(t, state, metric_fn, metric_fn_args)

    residuals_geodesic_vel_args = (pos_init, pos_end_targ, metric_fn,
                                   metric_fn_args, times, t_init, t_end,
                                   int_method)
    print('maxfev')
    print(maxfev)
    root_tol = 0.0005
    opt_result = root(
        residuals_geodesic_vel,
        vel_init_guess,
        args=residuals_geodesic_vel_args,
        # jac=loss_jac,
        options={'maxfev': maxfev},
        tol=root_tol)
    print('Root finding terminated...')
    print(opt_result)
    opt_vel_init = opt_result.x

    state_init = np.concatenate([pos_init, opt_vel_init])
    _, geodesic_traj = integrate_ode_fn(ode_fn,
                                        state_init,
                                        times,
                                        t_init=t_init,
                                        t_end=t_end,
                                        int_method=int_method)
    return opt_vel_init, geodesic_traj
