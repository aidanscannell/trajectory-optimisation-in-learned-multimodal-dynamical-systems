import time

import matplotlib.pyplot as plt
# import objax
from jax import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# from tromp.collocation import collocation
from tromp.helpers import (init_straight_trajectory,
                           init_svgp_gpjax_from_mogpe_ckpt)
from tromp.metric_tensors import SVGPMetricTensor
from tromp.ode import GeodesicODE
from tromp.plotting.solver import plot_solver_trajs_over_svgp
from tromp.plotting.trajectories import plot_trajs_over_svgp
from tromp.solvers import CollocationGeodesicSolver

#########################
# Configure solver params
#########################
maxiter = 500  # max number of iterations
num_col_points = 10  # number of collocation points to use in solver
lb_defect = -0.05
ub_defect = 0.05
lb_defect = -0.01  # works with cov=40 jitter=1e-4
ub_defect = 0.01
# lb_defect = -0.006
# ub_defect = 0.006
# lb_defect = -0.1
# ub_defect = 0.1
# lb_defect = -0.06
# ub_defect = 0.06
# lb_defect = -0.08
# ub_defect = 0.08
# lb_defect = -0.04  # works with cov=10 jitter=1e-4
# ub_defect = 0.04

# num_col_points = 10  # number of collocation points to use in solver
# lb_defect = -0.05  # works with cov=3.0 jitter=1e-4
# ub_defect = 0.05
# covariance_weight = 1.0
# jitter = 1e-4

# num_col_points = 12  # number of collocation points to use in solver
# lb_defect = -0.4  # works with cov=3.0 jitter=1e-4
# ub_defect = 0.4
# covariance_weight = 0.0
# jitter = 1e-4

# lb_defect = -0.000001
# ub_defect = 0.000001

# Initialise solver times
t_init = -1.0  # start time
t_end = 1.0  # end time
times = np.linspace(t_init, t_end, num_col_points)

# Initial guess of state vector at collocation points (guess straight line traj)
pos_init = np.array([2.7, 2.0])  # desired start state
pos_end_targ = np.array([-2.6, -1.5])  # desired end state
vel_init_guess = np.array([0.0000005, 0.0000003])  # initial guess of velocity
# vel_init_guess = np.array([0.0005, 0.0003])  # initial guess of velocity
state_guesses = init_straight_trajectory(
    pos_init, pos_end_targ, vel_init_guess, num_col_points=num_col_points
)


################################
# Configure metric tensor params
################################
# covariance_weight = 10.0
# covariance_weight = 100.0
# covariance_weight = 50.0
covariance_weight = 40.0
# covariance_weight = 20.0
# covariance_weight = 10.0
# covariance_weight = 30.0
# covariance_weight = 3.0
# covariance_weight = 7.0
# covariance_weight = 0.0
jitter = 1e-4
# jitter = 1e-6
# jitter = 1e-3
# jitter = 1e-2
# covariance_weight = 1.0
# # jitter = 1e-6
# jitter = 1e-8

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
    gp=svgp, covariance_weight=covariance_weight, jitter=jitter
)
ode = GeodesicODE(metric_tensor=metric_tensor)

collocation_solver = CollocationGeodesicSolver(
    ode=ode,
    covariance_weight=covariance_weight,
    maxiter=maxiter,
)


# plot_svgp_and_start_end(svgp, traj_init=state_guesses)
plot_trajs_over_svgp(svgp, traj_init=state_guesses)
plt.show()

t = time.time()
geodesic_traj = collocation_solver.solve_trajectory(
    state_guesses=state_guesses,
    pos_init=pos_init,
    pos_end_targ=pos_end_targ,
    times=times,
    lb_defect=lb_defect,
    ub_defect=ub_defect,
)
duration = time.time() - t
print("Optimisation duration: ", duration)


save_img_dir = "./images"

plot_solver_trajs_over_svgp(collocation_solver)
plt.savefig(
    save_img_dir + "/init-and-opt-trajs-on-svgp-new.pdf", transparent=True
)
plt.show()
traj_save_dir = "./saved_trajectories/opt_traj.npy"
np.save(traj_save_dir, geodesic_traj)
