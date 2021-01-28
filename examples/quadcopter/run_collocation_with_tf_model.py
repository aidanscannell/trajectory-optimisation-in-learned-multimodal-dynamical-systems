import matplotlib.pyplot as plt
import objax
from jax import numpy as np
from jax.config import config
from tromp.collocation import collocation
from tromp.helpers import (init_straight_trajectory,
                           init_svgp_gpjax_from_mogpe_ckpt)
from tromp.metric_tensors import SVGPMetricTensor
from tromp.solvers import CollocationGeodesicSolver, GeodesicODE
from tromp.visualisation.plot_trajectories import plot_svgp_and_start_end

config.update("jax_enable_x64", True)


mogpe_dir = "../../../mogpe/"
# ckpt_dir = mogpe_dir + "examples/logs/quadcopter/two_experts/09-08-144846"
ckpt_dir = mogpe_dir + "examples/logs/quadcopter/two_experts/11-16-101305"
config_file = mogpe_dir + "examples/quadcopter/configs/config_2_experts.toml"
data_file = mogpe_dir + "examples/quadcopter/data/quadcopter_data.npz"

svgp = init_svgp_gpjax_from_mogpe_ckpt(
    config_file, ckpt_dir, data_file, expert_num=1
)
gp = svgp

# How may collocation points to use in solver
num_col_points = 10

# Initialise solver times
t_init = -1.0
t_end = 1.0
times = np.linspace(t_init, t_end, num_col_points)

# Specify the desired start and end states
pos_init = np.array([0.7, -2.3])
pos_end_targ = np.array([1.0, 2.2])

# Initial guess of state vector at collocation points
# Guess a straight line trajectory
vel_init_guess = np.array([0.0000005, 0.0000003])
state_guesses = init_straight_trajectory(
    pos_init, pos_end_targ, vel_init_guess
)


# covariance_weight = 10.0
covariance_weight = 20.0
covariance_weight = 0.5
jitter = 1e-4
maxiter = 100

metric_tensor = SVGPMetricTensor(
    gp, covariance_weight=covariance_weight, jitter=jitter
)
ode = GeodesicODE(metric_tensor=metric_tensor)

collocation_solver = CollocationGeodesicSolver(
    ode=ode,
    num_col_points=num_col_points,
    covariance_weight=covariance_weight,
    maxiter=maxiter,
)


class CollocationSolverParams:
    times = times
    pos_init = pos_init
    pos_end_targ = pos_end_targ
    state_guesses = state_guesses


solver_params = CollocationSolverParams()
plot_svgp_and_start_end(gp, solver_params, traj_opts=None, labels=None)
plt.show()

geodesic_traj = collocation_solver.solve_trajectory(
    state_guesses=state_guesses,
    pos_init=pos_init,
    pos_end_targ=pos_end_targ,
    times=times,
)

plot_svgp_and_start_end(gp, solver_params, traj_opts=geodesic_traj)
plt.show()


# class SVGPMetric:
#     # gp = SVGP()
#     gp = svgp
#     # cov_weight = 0.5
#     cov_weight = 10.0
#     full_cov = True
#     jitter = 1e-4
#     metric_fn_kwargs = {
#         "X": gp.inducing_variable,
#         "kernel": gp.kernel,
#         "mean_func": gp.mean_function,
#         "f": gp.q_mu,
#         "full_cov": full_cov,
#         "q_sqrt": gp.q_sqrt,
#         "cov_weight": cov_weight,
#         "jitter": jitter,
#         "white": gp.whiten,
#     }
# metric_fn_kwargs = {
#     "X": gp.Z,
#     "kernel": gp.kernel,
#     "mean_func": gp.mean_func,
#     "f": gp.q_mu,
#     "full_cov": full_cov,
#     "q_sqrt": gp.q_sqrt,
#     "cov_weight": cov_weight,
#     "jitter": jitter,
#     "white": white,
# }


# metric = SVGPMetric()
# solver = CollocationSolverParams()
# plot_svgp_and_start_end(gp, solver, traj_opts=None, labels=None)

# plt.show()

# geodesic_traj = collocation(
#     state_guesses=state_guesses,
#     pos_init=pos_init,
#     pos_end_targ=pos_end_targ,
#     metric_fn=gp_metric_tensor,
#     metric_fn_kwargs=metric.metric_fn_kwargs,
#     times=times,
# )

# plot_svgp_and_start_end(gp, solver, traj_opts=geodesic_traj)
# plt.show()

# # geodesic_traj = collocation_root(solver.state_guesses, solver.pos_init,
# #                                  solver.pos_end_targ, metric_fn,
# #                                  metric.metric_fn_kwargs, solver.times)
# # geodesic_traj = collocation_(solver.pos_init, solver.pos_end_targ,
# #                              metric_fn, metric.metric_fn_kwargs,
# #                              solver.times)
