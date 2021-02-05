import time

import matplotlib.pyplot as plt
# import objax
from jax import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# from tromp.collocation import collocation
from tromp.helpers import (init_straight_trajectory,
                           init_svgp_gpjax_from_mogpe_ckpt)
from tromp.metric_tensors import SVGPMetricTensor
from tromp.ode import GeodesicODE
from tromp.plotting.plot_trajectories import plot_svgp_and_start_end
from tromp.plotting.solver import plot_solver_trajs_over_svgp
from tromp.solvers import CollocationGeodesicSolver

#########################
# Configure solver params
#########################
maxiter = 500  # max number of iterations
maxiter = 10  # max number of iterations
num_col_points = 10  # number of collocation points to use in solver

# Initialise solver times
t_init = -1.0  # start time
t_end = 1.0  # end time
times = np.linspace(t_init, t_end, num_col_points)

# Initial guess of state vector at collocation points (guess straight line traj)
pos_init = np.array([2.7, 2.0])  # desired start state
pos_end_targ = np.array([-2.6, -1.5])  # desired end state
vel_init_guess = np.array([0.05, 0.03])  # initial guess of velocity
# vel_init_guess = np.array([0.0005, 0.0003])  # initial guess of velocity
state_guesses = init_straight_trajectory(
    pos_init, pos_end_targ, vel_init_guess, num_col_points=num_col_points
)


################################
# Configure metric tensor params
################################
covariance_weight = 40.0
# covariance_weight = 20.0
# covariance_weight = 1.0
jitter_ode = 1e-6
# jitter_ode = 1e-9
jitter_metric = 1e-4

######################################################
# Load a SVGP gating function from an mogpe checkpoint
######################################################
expert_num = 1
mogpe_dir = "../../../../mogpe/"
ckpt_dir = (
    mogpe_dir
    + "examples/logs/quadcopter-sim/scenario-1/two_experts/01-19-121349"
)
config_file = (
    mogpe_dir
    + "examples/quadcopter-sim/scenario-1/configs/config_2_experts.toml"
)
data_file = (
    mogpe_dir
    + "examples/quadcopter-sim/scenario-1/data/quad_sim_const_action_scenario_1.npz"
)
svgp = init_svgp_gpjax_from_mogpe_ckpt(
    config_file, ckpt_dir, data_file, expert_num=expert_num
)


metric_tensor = SVGPMetricTensor(
    gp=svgp, covariance_weight=covariance_weight, jitter=jitter_metric
)
ode = GeodesicODE(metric_tensor=metric_tensor, jitter=jitter_ode)

collocation_solver = CollocationGeodesicSolver(
    ode=ode,
    covariance_weight=covariance_weight,
    maxiter=maxiter,
)


# plot_svgp_and_start_end(svgp, traj_init=state_guesses)
# plt.show()

t = time.time()
geodesic_traj = collocation_solver.solve_trajectory_lagrange(
    state_guesses=state_guesses,
    pos_init=pos_init,
    pos_end_targ=pos_end_targ,
    times=times,
)
duration = time.time() - t
print("Optimisation duration: ", duration)


save_img_dir = "./images"
# plot_svgp_and_start_end(svgp, traj_init=state_guesses, traj_opts=geodesic_traj)
plot_solver_trajs_over_svgp(collocation_solver)
plt.savefig(
    save_img_dir + "/init-and-opt-trajs-on-svgp-unconstrained.pdf", transparent=True
)
plt.show()
traj_save_dir = "./saved_trajectories/opt_traj_unconstrained.npy"
np.save(traj_save_dir, geodesic_traj)
