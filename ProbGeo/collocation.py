import jax
import jax.numpy as np
import scipy
from jax.config import config
from scipy.optimize import Bounds, NonlinearConstraint

from ProbGeo.ode import geodesic_ode

config.update("jax_enable_x64", True)


def collocation_objective(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    # state_guesses = state_guesses[:-1]
    # times[-1] = state_guesses[-1]
    # times = jax.ops.index_update(times, jax.ops.index[-1], state_guesses[-1])
    # defect = collocation_constraint_fn(state_guesses, pos_init, pos_end_targ,
    #                                    metric_fn, metric_fn_kwargs, times)
    # sum_defect = np.sum(defect)
    # print('sum')
    # print(sum_defect.shape)
    # return sum_defect
    input_dim = 2
    state_guesses = state_guesses.reshape([-1, 2 * input_dim])
    pos_guesses = state_guesses[:, 0:2]
    pos_guesses = jax.ops.index_update(
        pos_guesses, jax.ops.index[0, :], pos_init
    )
    pos_guesses = jax.ops.index_update(
        pos_guesses, jax.ops.index[-1, :], pos_end_targ
    )
    norm = np.linalg.norm(pos_guesses, axis=-1, ord=-2)
    print("norm")
    print(norm.shape)
    # state_guesses = jax.ops.index_update(state_guesses,
    #                                      jax.ops.index[0,
    #                                                    0:input_dim], pos_init)
    # state_guesses = jax.ops.index_update(state_guesses,
    #                                      jax.ops.index[-1, 0:input_dim],
    #                                      pos_end_targ)

    # fun = metric_fn_kwargs['fun']
    # fun_kwargs = metric_fn_kwargs['fun_kwargs']
    # # calculate metric tensor and jacobian
    # metric_tensor, jac = metric_fn(pos_guesses, fun, fun_kwargs)

    from ProbGeo.metric_tensor import gp_metric_tensor

    # fun_kwargs = metric_fn_kwargs['fun_kwargs']
    # metric_tensor = gp_metric_tensor()
    # metric_tensor, _, _ = metric_fn(pos_guesses, **metric_fn_kwargs)
    try:
        metric_tensor, _ = metric_fn(pos_guesses, **metric_fn_kwargs)
    except:
        metric_tensor, _, _ = metric_fn(pos_guesses, **metric_fn_kwargs)
    print("metric_tensor")
    print(metric_tensor.shape)
    # metric_tensor, jac = metric_fn(pos_guesses, **metric_fn_kwargs)
    trace_metric = np.trace(metric_tensor, axis1=-2, axis2=-1)
    print(trace_metric.shape)
    trace_metric_sum = np.sum(trace_metric)

    # norm = np.linalg.norm(state_guesses, axis=-1)
    # trace_metric_sum = 10 * trace_metric_sum
    norm_sum = np.sum(norm)
    print("Norm Loss: ", norm_sum)
    print("Trace Metric Loss: ", trace_metric_sum)
    print("Time Loss: ", times[-1])
    # print(state_guesses)
    # return 1.
    # return times[-1]
    # return norm_sum
    # return trace_metric_sum
    metric_weight = 5.0
    metric_weight = 50.0
    return norm_sum + metric_weight * trace_metric_sum


def collocation(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    def collocation_constraint_fn(state_guesses):
        # state_guesses = state_guesses[:-1]
        input_dim = 2
        state_guesses = state_guesses.reshape([-1, 2 * input_dim])
        num_timesteps = times.shape[0]
        dt = times[-1] - times[0]
        time_col = np.linspace(
            times[0] + dt / 2, times[-1] - dt / 2, num_timesteps - 1
        )

        def ode_fn(state):
            return geodesic_ode(times, state, metric_fn, metric_fn_kwargs)

        state_prime = jax.vmap(ode_fn)(state_guesses)
        state_ll = state_guesses[0:-1, :]
        state_rr = state_guesses[1:, :]
        state_prime_ll = state_prime[0:-1, :]
        state_prime_rr = state_prime[1:, :]
        state_col = 0.5 * (state_ll + state_rr) + dt / 8 * (
            state_prime_ll - state_prime_rr
        )
        # print('state col')
        # print(state_col.shape)
        state_prime_col = jax.vmap(ode_fn)(state_col)
        # print('state prime col')
        # print(state_prime_col.shape)

        defect = (state_ll - state_rr) + dt / 6 * (
            state_prime_ll + 4 * state_prime_col + state_prime_rr
        )
        print("defect")
        print(defect)
        # print(defect.shape)
        return defect.flatten()

    # loss_args = (pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times)
    loss_args = (pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times)
    # defect_constraints = NonlinearConstraint(collocation_constraint_fn, -0.01,
    #                                          0.01)
    # defect_constraints = NonlinearConstraint(collocation_constraint_fn, -0.05,
    #                                          0.05)
    len_constraints = state_guesses.flatten().shape[0]
    print("state")
    print(state_guesses.shape)
    print("len_constraints")
    print(len_constraints)
    lb = -np.ones([len_constraints]) * np.inf
    ub = np.ones([len_constraints]) * np.inf
    print(lb.shape)
    print(ub.shape)
    # defect_constraints = NonlinearConstraint(collocation_constraint_fn, lb, ub)
    # defect_constraints = NonlinearConstraint(collocation_constraint_fn,
    #                                          -np.inf, np.inf)
    # defect_constraints = NonlinearConstraint(collocation_constraint_fn, -0.1,
    #                                          0.1)
    # defect_constraints = NonlinearConstraint(collocation_constraint_fn, -0.01,
    #                                          0.01)
    defect_constraints = NonlinearConstraint(
        collocation_constraint_fn, -0.1, 0.1
    )
    # defect_constraints = NonlinearConstraint(collocation_constraint_fn,
    # -10000., 10000.)
    # bounds = init_bounds(state_guesses, pos_init)

    bounds = start_end_pos_bounds(state_guesses, pos_init, pos_end_targ)

    constraints = defect_constraints
    # method = "L-BFGS-B"
    # method = "CG"
    # method = "TNC"
    method = "SLSQP"
    print("times")
    print(times[-1].shape)
    print(state_guesses.flatten().shape)
    params = np.concatenate([state_guesses.flatten(), times[-1].reshape([-1])])
    print("prams")
    print(params.shape)
    res = scipy.optimize.minimize(
        collocation_objective,
        state_guesses.flatten(),
        # params,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"disp": True, "maxiter": 100},
        args=loss_args,
    )
    print("res")
    print(res)
    print(res.x.shape)
    state_opt = res.x
    # state_opt = res.x[:-1]
    state_opt = state_opt.reshape([*state_guesses.shape])
    return state_opt


def collocation_objective_inc_constraints(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    def collocation_constraint_fn(state_guesses):
        input_dim = 2
        state_guesses = state_guesses.reshape([-1, 2 * input_dim])
        num_timesteps = times.shape[0]
        dt = times[-1] - times[0]
        time_col = np.linspace(
            times[0] + dt / 2, times[-1] - dt / 2, num_timesteps - 1
        )

        def ode_fn(state):
            return geodesic_ode(times, state, metric_fn, metric_fn_kwargs)

        state_prime = jax.vmap(ode_fn)(state_guesses)
        state_ll = state_guesses[0:-1, :]
        state_rr = state_guesses[1:, :]
        state_prime_ll = state_prime[0:-1, :]
        state_prime_rr = state_prime[1:, :]
        state_col = 0.5 * (state_ll + state_rr) + dt / 8 * (
            state_prime_ll - state_prime_rr
        )
        # print('state col')
        # print(state_col.shape)
        state_prime_col = jax.vmap(ode_fn)(state_col)
        # print('state prime col')
        # print(state_prime_col.shape)

        defect = (state_ll - state_rr) + dt / 6 * (
            state_prime_ll + 4 * state_prime_col + state_prime_rr
        )
        # print('defect')
        # print(defect.shape)
        return defect.flatten()

    defect = collocation_constraint_fn(state_guesses)
    defect_squared = np.sqrt(defect ** 2)
    sum_defect = np.sum(defect_squared)
    # defect_loss

    input_dim = 2
    state_guesses = state_guesses.reshape([-1, 2 * input_dim])
    pos_guesses = state_guesses[:, 0:2]
    pos_guesses = jax.ops.index_update(
        pos_guesses, jax.ops.index[0, :], pos_init
    )
    pos_guesses = jax.ops.index_update(
        pos_guesses, jax.ops.index[-1, :], pos_end_targ
    )
    norm = np.linalg.norm(pos_guesses, axis=-1)
    norm_sum = np.sum(norm)
    # loss = norm_sum + sum_defect
    loss = sum_defect
    print("Loss: ", loss)
    return loss


def collocation_no_constraint(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    print("inside no consttaint")

    loss_args = (pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times)
    lb = -np.ones([*state_guesses.shape]) * np.inf
    ub = np.ones([*state_guesses.shape]) * np.inf

    for idx, pos in enumerate(pos_init):
        if pos < 0:
            lb = jax.ops.index_update(lb, jax.ops.index[0, idx], pos * 1.02)
            ub = jax.ops.index_update(ub, jax.ops.index[0, idx], pos * 0.98)
        else:
            lb = jax.ops.index_update(lb, jax.ops.index[0, idx], pos * 0.98)
            ub = jax.ops.index_update(ub, jax.ops.index[0, idx], pos * 1.02)

    for idx, pos in enumerate(pos_end_targ):
        if pos < 0:
            lb = jax.ops.index_update(lb, jax.ops.index[-1, idx], pos * 1.02)
            ub = jax.ops.index_update(ub, jax.ops.index[-1, idx], pos * 0.98)
        else:
            lb = jax.ops.index_update(lb, jax.ops.index[-1, idx], pos * 0.98)
            ub = jax.ops.index_update(ub, jax.ops.index[-1, idx], pos * 1.02)

    bounds = Bounds(lb=lb.flatten(), ub=ub.flatten())
    # method = "L-BFGS-B"
    # method = "CG"
    # method = "TNC"
    method = "SLSQP"
    res = scipy.optimize.minimize(
        collocation_objective_inc_constraints,
        state_guesses.flatten(),
        method=method,
        bounds=bounds,
        # constraints=constraints,
        options={"disp": True, "maxiter": 100},
        args=loss_args,
    )
    print("res")
    print(res)
    print(res.x.shape)
    state_opt = res.x.reshape([*state_guesses.shape])
    return state_opt


def collocation_defect(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    input_dim = 2
    state_guesses = state_guesses.reshape([-1, 2 * input_dim])
    state_guesses = jax.ops.index_update(
        state_guesses, jax.ops.index[0, 0:input_dim], pos_init
    )
    state_guesses = jax.ops.index_update(
        state_guesses, jax.ops.index[-1, 0:input_dim], pos_end_targ
    )
    num_timesteps = times.shape[0]
    dt = times[-1] - times[0]
    time_col = np.linspace(
        times[0] + dt / 2, times[-1] - dt / 2, num_timesteps - 1
    )

    def ode_fn(state):
        return geodesic_ode(times, state, metric_fn, metric_fn_kwargs)

    state_prime = jax.vmap(ode_fn)(state_guesses)
    state_ll = state_guesses[0:-1, :]
    state_rr = state_guesses[1:, :]
    state_prime_ll = state_prime[0:-1, :]
    state_prime_rr = state_prime[1:, :]
    state_col = 0.5 * (state_ll + state_rr) + dt / 8 * (
        state_prime_ll - state_prime_rr
    )
    # print('state col')
    # print(state_col.shape)
    state_prime_col = jax.vmap(ode_fn)(state_col)
    # print('state prime col')
    # print(state_prime_col.shape)

    defect = (state_ll - state_rr) + dt / 6 * (
        state_prime_ll + 4 * state_prime_col + state_prime_rr
    )
    zeros = np.zeros([1, defect.shape[1]])
    # print('defect')
    # print(defect)
    defect = np.concatenate([zeros, defect])
    return defect.flatten()


def collocation_defect_jac(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):

    jac_defect = jax.jacfwd(collocation_defect)(
        state_guesses,
        pos_init,
        pos_end_targ,
        metric_fn,
        metric_fn_kwargs,
        times,
    )
    return jac_defect


def collocation_defect_and_jac(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    jac_defect = collocation_defect_jac(
        state_guesses,
        pos_init,
        pos_end_targ,
        metric_fn,
        metric_fn_kwargs,
        times,
    )
    defect = collocation_defect(
        state_guesses,
        pos_init,
        pos_end_targ,
        metric_fn,
        metric_fn_kwargs,
        times,
    )
    return defect, jac_defect


def collocation_root(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    print("inside collocation_root")
    input_dim = pos_init.shape[0]
    jac_defect = collocation_defect_jac(
        state_guesses.flatten(),
        pos_init,
        pos_end_targ,
        metric_fn,
        metric_fn_kwargs,
        times,
    )
    print("jac")

    args = (pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times)
    method = "hybr"
    # method = "lm"
    tol = 0.0000005
    # tol = 0.5
    res = scipy.optimize.root(
        collocation_defect,
        # collocation_defect_and_jac,
        state_guesses.flatten(),
        method=method,
        tol=tol,
        # options={
        #     'disp': True,
        #     'maxiter': 100
        # },
        # jac=True,
        args=args,
    )
    print("res")
    print(res)
    print(res.x.shape)
    state_opt = res.x.reshape([*state_guesses.shape])

    state_opt = jax.ops.index_update(
        state_opt, jax.ops.index[0, 0:input_dim], pos_init
    )
    state_opt = jax.ops.index_update(
        state_opt, jax.ops.index[-1, 0:input_dim], pos_end_targ
    )
    return state_opt


def calc_differentiator_matrix(col_times, pol_deg=3):
    num_col = pol_deg + 1
    # degree of polynominal = number of colloc points
    # matrix of powers for col_times
    # power_matrix = np.arange(1., num_col + 1, 1)
    power_matrix = np.arange(0.0, num_col, 1)
    collocation_power_matrix = np.tile(power_matrix, [num_col, 1])
    # print('power matrix')
    # print(power_matrix)
    # print(power_matrix.shape)
    # print('collocation_power matrix')
    # print(collocation_power_matrix)
    # print(collocation_power_matrix.shape)

    # matrix of broadcast collocation times
    col_times_matrix = col_times.T * np.ones([1, num_col])
    # print("col_times_matrix")
    # print(col_times_matrix)
    # print(col_times_matrix.shape)

    # vandermonde matrix (polynominal times without constants)
    vandermonde = col_times_matrix ** collocation_power_matrix
    inv_vandermonde = np.linalg.inv(vandermonde)
    diff_vandermonde = collocation_power_matrix * (
        col_times_matrix ** (collocation_power_matrix - 1)
    )
    # print('vandermonde')
    # print(vandermonde)
    # print("diff_vandermonde")
    # print(diff_vandermonde)
    # inv_diff_vandermonde = np.linalg.inv(diff_vandermonde)
    # print("inv_diff_vandermonde")
    # print(inv_diff_vandermonde)

    # collocation matrix (xdot = differentiator_matrix @ x)
    differentiator_matrix = diff_vandermonde @ inv_vandermonde
    # differentiator_matrix = vandermonde @ inv_diff_vandermonde
    print("differentiator_matrix")
    print(differentiator_matrix)
    return differentiator_matrix


def init_lobatto_col_times(pol_deg, end_time=1.0):
    # define collocation points from Lobatto quadrature
    num_col = pol_deg + 1
    # collocation points - t is generalized time, [-1,1]
    col_times = (
        np.cos(np.arange(0, num_col, 1) * np.pi / (num_col - 1))
    ).reshape(1, -1)
    print("col_times")
    print(col_times)
    return col_times
    # if (num_col == 2):
    #     col_times = np.array([0.0, 1.0])
    # elif (num_col == 3):
    #     col_times = np.array([0.0, 0.5, 1.0])
    # elif (num_col == 4):
    #     col_times = np.array(
    #         [0.0, 0.5 - np.sqrt(5) / 10.0, 0.5 + np.sqrt(5) / 10.0, 1.0])
    # elif (num_col == 5):
    #     col_times = np.array(
    #         [0.0, 0.5 - np.sqrt(21) / 14.0, 0.5, 0.5 + np.sqrt(21) / 14.0, 1])
    # elif (num_col == 6):
    #     col_times = np.array([
    #         0.0, 0.5 - np.sqrt((7.0 + 2.0 * np.sqrt(7.0)) / 21.0) / 2.0,
    #         0.5 - np.sqrt(
    #             (7.0 - 2.0 * np.sqrt(7.0)) / 21.0) / 2.0, 0.5 + np.sqrt(
    #                 (7.0 - 2.0 * np.sqrt(7.0)) / 21.0) / 2.0, 0.5 + np.sqrt(
    #                     (7.0 + 2.0 * np.sqrt(7.0)) / 21.0) / 2.0, 1.0
    #     ])
    # col_times = col_times.reshape(1, -1)
    # return col_times * end_time


def start_end_pos_bounds(state_guesses, pos_init, pos_end):
    lb = -np.ones([*state_guesses.shape]) * np.inf
    ub = np.ones([*state_guesses.shape]) * np.inf

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


def init_state_guesses(pos_init, pos_end, vel, num_elems=2, num_col_points=4):
    if len(pos_init.shape) == 1:
        input_dim = pos_init.shape[0]
    elif len(pos_init.shape) == 2:
        input_dim = pos_init.shape[1]
    # pos_guesses = []
    # for start_pos, end_pos in zip(pos_init, pos_end):
    #     pos_guesses.append(
    #         np.linspace(start_pos, end_pos, num_col_points * num_elems))
    # pos_guesses = np.stack(pos_guesses, -1)

    # pos_mid = np.array([-.5, 0.5])
    pos_mid = np.array([-0.5, 1.0])
    pos11_guesses = np.linspace(
        pos_init[0], pos_mid[0], int(num_col_points * num_elems / 2)
    )
    pos21_guesses = np.linspace(
        pos_init[1], pos_mid[1], int(num_col_points * num_elems / 2)
    )
    pos12_guesses = np.linspace(
        pos_mid[0], pos_end[0], int(num_col_points * num_elems / 2)
    )
    pos22_guesses = np.linspace(
        pos_mid[1], pos_end[1], int(num_col_points * num_elems / 2)
    )
    pos1_guesses = np.concatenate([pos11_guesses, pos12_guesses])
    pos2_guesses = np.concatenate([pos21_guesses, pos22_guesses])
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)

    vel_guesses = np.broadcast_to(vel, (num_col_points * num_elems, input_dim))

    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)
    return state_guesses


# def collocation_(state_guesses, pos_init, pos_end_targ, metric_fn,
#                  metric_fn_kwargs, times):
def collocation_(pos_init, pos_end, metric_fn, metric_fn_kwargs, times):
    def collocation_constraints(state_guesses):
        input_dim = 2
        state_guesses = state_guesses.reshape([-1, 2 * input_dim])
        state_guesses = jax.ops.index_update(
            state_guesses, jax.ops.index[0, 0:input_dim], pos_init
        )
        state_guesses = jax.ops.index_update(
            state_guesses, jax.ops.index[-1, 0:input_dim], pos_end
        )

        def ode_fn(state):
            return geodesic_ode(times, state, metric_fn, metric_fn_kwargs)

        print("state guesses")
        print(state_guesses.shape)

        state_prime = jax.vmap(ode_fn)(state_guesses)
        # state_prime = jax.vmap(ode_fn)(state_guesses[0:-1, :])
        print("state prime")
        print(state_prime.shape)

        state_prime_col = differentiator_matrix @ state_guesses.T
        # state_prime_col = differentiator_matrix @ state_guesses[0:-1, :].T
        # state_prime_col = jax.vmap(ode_fn)(state_col)
        print("sate_prime_col")
        print(state_prime_col.shape)
        defect = state_prime_col.T - state_prime

        print("defect")
        print(defect)
        print(defect.shape)
        print(defect.flatten().shape)
        return defect.flatten()

    pol_deg = 3
    num_elems = 1
    # num_elems = 3
    vel = np.array([0.0000005, 0.0000003])
    time_init = -1.0
    time_end = 1.0

    num_col_points = pol_deg + 1
    state_guesses = init_state_guesses(
        pos_init,
        pos_end,
        vel,
        num_elems=num_elems,
        num_col_points=num_col_points,
    )
    print("state guesses")
    print(state_guesses.shape)
    print(state_guesses)

    col_times = init_lobatto_col_times(pol_deg)
    differentiator_matrix = calc_differentiator_matrix(col_times, pol_deg)
    bounds = start_end_pos_bounds(state_guesses, pos_init, pos_end)

    print("col_times")
    print(col_times)
    print(col_times.shape)
    print("collocation matrix")
    print(differentiator_matrix)
    print(differentiator_matrix.shape)

    # loss_args = (pos_init, pos_end, metric_fn, metric_fn_kwargs, times)

    defect_constraints = NonlinearConstraint(
        collocation_constraints, -0.1, 0.1
    )
    constraints = defect_constraints
    # method = "L-BFGS-B"
    # method = "CG"
    # method = "TNC"
    # method = "SLSQP"

    method = "hybr"
    method = "lm"
    tol = 0.0000005
    # tol = 0.5
    res = scipy.optimize.root(
        collocation_constraints,
        # collocation_defect_and_jac,
        state_guesses.flatten(),
        method=method,
        tol=tol,
        # options={
        #     'disp': True,
        #     'maxiter': 100
        # },
        # jac=True,
    )
    # res = scipy.optimize.fsolve(collocation_constraints, state_guesses)
    # res = scipy.optimize.minimize(collocation_objective,
    #                               state_guesses.flatten(),
    #                               method=method,
    #                               bounds=bounds,
    #                               constraints=constraints,
    #                               options={
    #                                   'disp': True,
    #                                   'maxiter': 100
    #                               },
    #                               args=loss_args)
    print("res")
    print(res)
    # state_opt = res.reshape([*state_guesses.shape])
    # print(res)
    print(res.x.shape)
    state_opt = res.x.reshape([*state_guesses.shape])
    return state_opt

    # # construct matrix to evaluate values at endpoints etc

    # # constraint points
    # constraint_times = np.linspace(-1, 1, num_constraint_pts).reshape(
    #     1, num_constraint_pts)

    # # matrix of powers
    # con_power_matrix = np.tile(power_matrix, [num_constraint_pts, 1])
    # print('con_power_matrix')
    # print(con_power_matrix)
    # print(con_power_matrix.shape)

    # # matrix of colloc points
    # constraint_matrix = constraint_times.T * np.ones([1, num_col])
    # print('constraint_matrix')
    # print(constraint_matrix)

    # # vandermonde matrix
    # conVanDerMonde = constraint_matrix**con_power_matrix
    # print("conVanDerMonde")
    # print(conVanDerMonde)

    # # constraint evalution matrix
    # evalMatrix = conVanDerMonde @ inv_vandermonde
    # print('eval matrix')
    # print(evalMatrix)


def collocation_interpolation(
    state_guesses, pos_init, pos_end_targ, metric_fn, metric_fn_kwargs, times
):
    state_guesses = state_guesses[:-1]
    input_dim = 2
    state_guesses = state_guesses.reshape([-1, input_dim])
    num_timesteps = times.shape[0]
    dt = times[-1] - times[0]
    time_col = np.linspace(
        times[0] + dt / 2, times[-1] - dt / 2, num_timesteps - 1
    )

    def ode_fn(state):
        return geodesic_ode(times, state, metric_fn, metric_fn_kwargs)

    state_prime = jax.vmap(ode_fn)(state_guesses)
    state_ll = state_guesses[0:-1, :]
    state_rr = state_guesses[1:, :]
    state_prime_ll = state_prime[0:-1, :]
    state_prime_rr = state_prime[1:, :]
    state_col = 0.5 * (state_ll + state_rr) + dt / 8 * (
        state_prime_ll - state_prime_rr
    )
    # print('state col')
    # print(state_col.shape)
    state_prime_col = jax.vmap(ode_fn)(state_col)
    # print('state prime col')
    # print(state_prime_col.shape)

    defect = (state_ll - state_rr) + dt / 6 * (
        state_prime_ll + 4 * state_prime_col + state_prime_rr
    )


# collocation_()
