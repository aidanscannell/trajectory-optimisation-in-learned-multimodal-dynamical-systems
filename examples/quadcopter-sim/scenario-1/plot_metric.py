import matplotlib.pyplot as plt
from jax import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# from tromp.collocation import collocation
from tromp.helpers import (init_straight_trajectory,
                           init_svgp_gpjax_from_mogpe_ckpt)
from tromp.metric_tensors import SVGPMetricTensor
from tromp.visualisation.metric import plot_svgp_metric_trace

################################
# Configure metric tensor params
################################
covariance_weight = 40.0
# covariance_weight = 1.0
jitter_ode = 1e-4
jitter_ode = 1e-9
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


fig, axs = plot_svgp_metric_trace(metric_tensor)

# save_img_dir = "./images"
# plot_svgp_and_start_end(svgp, traj_init=state_guesses, traj_opts=geodesic_traj)
# plt.savefig(
#     save_img_dir + "/init-and-opt-trajs-on-svgp-new.pdf", transparent=True
# )
plt.show()
