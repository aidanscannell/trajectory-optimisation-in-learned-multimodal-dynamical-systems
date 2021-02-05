import jax
import matplotlib
import matplotlib.pyplot as plt
from jax.config import config
from tromp.visualisation.gp import plot_contourf, plot_svgp_mean_and_var
from tromp.visualisation.utils import create_grid

config.update("jax_enable_x64", True)

from jax import numpy as np

params = {
    "text.usetex": True,
    "text.latex.preamble": [
        "\\usepackage{amssymb}",
        "\\usepackage{amsmath}",
    ],
}
plt.rcParams.update(params)

color_init = "c"
color_opt = "m"
color_opt_1 = "m"
color_opt_2 = "olive"
color_opts = [color_opt_2, color_opt_1]

label_init = "Initial"
label_opt = "Optimised"
mean_label = "$\mathbb{E}[h^{(1)}]$"
var_label = "$\mathbb{V}[h^{(1)}]$"
x_label = "$x$"
y_label = "$y$"

def plot_solver_trajs_over_svgp(solver):
    svgp = solver.ode.metric_tensor.gp
    traj_init = solver.state_guesses
    traj_opt = solver.optimised_trajectory

    fig, axs = plot_svgp_mean_and_var(svgp, mean_label=mean_label, var_label=var_label)
    for ax in axs:
        fig, ax = plot_start_and_end_pos(
            fig, ax, pos_init=traj_init[0, :], pos_end=traj_init[-1, :]
        )
        plot_traj(fig, ax, traj_init, color=color_init, label=label_init)
        if traj_opt is not None:
            plot_traj(fig, ax, traj_opt, color=color_opt, label=label_opt)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.scatter(gp.inducing_variable[:, 0], gp.inducing_variable[:, 1])
    axs[0].legend(loc="lower left")
    return fig, axs


def plot_start_and_end_pos(fig, ax, pos_init, pos_end):
    ax.scatter(pos_init[0], pos_init[1], marker="x", color="k")
    ax.scatter(pos_end[0], pos_end[1], color="k", marker="x")
    ax.annotate(
        "Start $\mathbf{x}_0$",
        (pos_init[0], pos_init[1]),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.annotate(
        "End $\mathbf{x}_f$",
        (pos_end[0], pos_end[1]),
        horizontalalignment="left",
        verticalalignment="top",
    )
    return fig, ax


def plot_traj(fig, ax, traj, color="k", label=""):
    ax.scatter(traj[:, 0], traj[:, 1], marker="x", color=color)
    ax.plot(traj[:, 0], traj[:, 1], color=color, label=label)
    return fig, ax
