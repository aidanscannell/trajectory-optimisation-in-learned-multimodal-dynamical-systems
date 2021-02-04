import matplotlib.pyplot as plt
from jax import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
from tromp.helpers import (init_straight_trajectory,
                           init_svgp_gpjax_from_mogpe_ckpt)
from tromp.metric_tensors import SVGPMetricTensor
from tromp.visualisation.gp import (plot_svgp_jacobian_mean,
                                    plot_svgp_jacobian_var)

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

# fig, axs = plot_svgp_jacobian_mean(svgp)
fig, axs = plot_svgp_jacobian_var(svgp)
plt.show()
