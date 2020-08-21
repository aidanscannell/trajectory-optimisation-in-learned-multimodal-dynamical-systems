from jax import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from ProbGeo.ode import geodesic_ode


def integrate_ode_fn(ode_fn,
                     state_init,
                     times,
                     t_init=0.,
                     t_end=1.,
                     int_method='RK45'):
    integrator = solve_ivp(fun=ode_fn,
                           t_span=(t_init, t_end),
                           y0=state_init,
                           t_eval=times,
                           method=int_method)
    return integrator.t, integrator.y


def error_geodesic_vel(vel_init,
                       pos_init,
                       pos_end_targ,
                       metric_fn,
                       metric_fn_args,
                       times,
                       t_init=0.,
                       t_end=1.,
                       int_method='RK45'):
    def ode_fn(t, state):
        return geodesic_ode(t, state, metric_fn, metric_fn_args)

    input_dim = pos_init.shape[0]
    print('Shooting with initial velocity: ', vel_init)
    state_init = np.concatenate([pos_init, vel_init])

    ts, states = integrate_ode_fn(ode_fn,
                                  state_init,
                                  times,
                                  t_init=t_init,
                                  t_end=t_end,
                                  int_method=int_method)
    pos_end_integrated = np.array(states)[0:input_dim, -1]

    error = pos_end_targ - pos_end_integrated
    print('Target pos error: ', error)
    return error


def shooting_geodesic_solver(pos_init,
                             pos_end_targ,
                             vel_init_guess,
                             metric_fn,
                             metric_fn_kwargs,
                             times,
                             t_init=0.,
                             t_end=1.,
                             int_method='RK45',
                             root_tol=0.05,
                             maxfev=1000):
    def ode_fn(t, state):
        return geodesic_ode(t, state, metric_fn, metric_fn_kwargs)

    error_geodesic_vel_args = (pos_init, pos_end_targ, metric_fn,
                               metric_fn_kwargs, times, t_init, t_end,
                               int_method)
    opt_result = root(
        error_geodesic_vel,
        vel_init_guess,
        args=error_geodesic_vel_args,
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
