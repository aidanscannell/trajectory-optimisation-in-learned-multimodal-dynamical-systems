import jax
import matplotlib.pyplot as plt
from gpjax.prediction import gp_predict
from jax import numpy as np
from jax.config import config
from matplotlib import cm
from ProbGeo.metric_tensor import gp_metric_tensor
from ProbGeo.mogpe import single_mogpe_mixing_probability
from ProbGeo.visualisation.utils import create_grid

config.update("jax_enable_x64", True)

params = {
    # 'axes.labelsize': 30,
    # 'font.size': 30,
    # 'legend.fontsize': 20,
    # 'xtick.labelsize': 30,
    # 'ytick.labelsize': 30,
    "text.usetex": True,
    # 'text.latex.preamble': ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
}
plt.rcParams.update(params)
color_init = "c"
color_opt = "m"


def plot_3d_surf(fig, ax, x, y, z):
    surf = ax.plot_surface(
        x,
        y,
        z.reshape([*x.shape]),
        rstride=1,
        cstride=1,
        alpha=0.7,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.elev = 30
    ax.azim = -41
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return fig, ax


def plot_3d_mean_and_var(gp, solver):

    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(
        Xnew,
        gp.Z,
        kernels=gp.kernel,
        mean_funcs=gp.mean_func,
        f=gp.q_mu,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_mu = fig.add_subplot(1, 2, 1, projection="3d")
    surf_mu = plot_3d_surf(fig, ax_mu, xx, yy, mu)
    ax_var = fig.add_subplot(1, 2, 2, projection="3d")
    surf_var = plot_3d_surf(fig, ax_var, xx, yy, var)

    axs = [ax_mu, ax_var]
    ax_mu.set_zlabel("Mean")
    ax_var.set_zlabel("Variance")
    return fig, axs


def plot_3d_metric_trace(metric, solver):
    # plot original GP
    Xnew, xx, yy = create_grid(metric.gp.X, N=961)
    metric_tensor, mu_j, cov_j = gp_metric_tensor(
        Xnew,
        metric.gp.Z,
        metric.gp.kernel,
        mean_func=metric.gp.mean_func,
        f=metric.gp.q_mu,
        full_cov=True,
        q_sqrt=metric.gp.q_sqrt,
        cov_weight=metric.cov_weight,
    )

    metric_trace = np.trace(metric_tensor, axis1=1, axis2=2)
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surf = plot_3d_surf(fig, ax, xx, yy, metric_trace)
    ax.set_zlabel("Tr$(G(\mathbf{x}_*))$")

    return fig, ax


def plot_3d_mixing_prob(gp, solver):

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)

    mixing_probs = jax.vmap(
        single_mogpe_mixing_probability,
        (0, None, None, None, None, None, None),
    )(Xnew, gp.Z, gp.kernel, gp.mean_func, gp.q_mu, False, gp.q_sqrt)

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surf = plot_3d_surf(fig, ax, xx, yy, mixing_probs[:, 0:1])
    ax.set_zlabel("$\Pr(\\alpha_*=0 | \mathbf{x}_*)$")

    return fig, ax


def plot_3d_traj_mean_and_var(fig, axs, gp, traj):
    Xnew, xx, yy = create_grid(gp.X, N=961)
    traj_mu, traj_var = gp_predict(
        traj[:, 0:2],
        gp.Z,
        kernels=gp.kernel,
        mean_funcs=gp.mean_func,
        f=gp.q_mu,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    plot_3d_traj(fig, axs[0], traj, zs=traj_mu)
    plot_3d_traj(fig, axs[1], traj, zs=traj_var)
    return fig, axs


def plot_3d_traj_metric_trace(fig, ax, metric, traj):
    metric_tensor, mu_j, cov_j = gp_metric_tensor(
        traj[:, 0:2],
        metric.gp.Z,
        metric.gp.kernel,
        mean_func=metric.gp.mean_func,
        f=metric.gp.q_mu,
        full_cov=True,
        q_sqrt=metric.gp.q_sqrt,
        cov_weight=metric.cov_weight,
    )

    metric_trace = np.trace(metric_tensor, axis1=1, axis2=2)

    plot_3d_traj(fig, ax, traj, zs=metric_trace)
    return fig, ax


def plot_3d_traj_mixing_prob(fig, ax, gp, traj):

    mixing_probs = jax.vmap(
        single_mogpe_mixing_probability,
        (0, None, None, None, None, None, None),
    )(traj[:, 0:2], gp.Z, gp.kernel, gp.mean_func, gp.q_mu, False, gp.q_sqrt)

    plot_3d_traj(fig, ax, traj, zs=mixing_probs[:, 0:1])
    return fig, ax


def plot_3d_traj_mixing_prob_init_and_opt(fig, ax, gp, traj_init, traj_opt):

    mixing_probs_init = jax.vmap(
        single_mogpe_mixing_probability,
        (0, None, None, None, None, None, None),
    )(
        traj_init[:, 0:2],
        gp.Z,
        gp.kernel,
        gp.mean_func,
        gp.q_mu,
        False,
        gp.q_sqrt,
    )
    mixing_probs_opt = jax.vmap(
        single_mogpe_mixing_probability,
        (0, None, None, None, None, None, None),
    )(
        traj_opt[:, 0:2],
        gp.Z,
        gp.kernel,
        gp.mean_func,
        gp.q_mu,
        False,
        gp.q_sqrt,
    )

    plot_3d_traj_col(
        fig, ax, traj_init, zs=mixing_probs_init[:, 0:1], color="c"
    )
    plot_3d_traj_col(fig, ax, traj_opt, zs=mixing_probs_opt[:, 0:1], color="m")
    return fig, ax


def plot_3d_traj_metric_trace_init_and_opt(
    fig, ax, metric, traj_init, traj_opt
):

    metric_tensor_init, _, _ = gp_metric_tensor(
        traj_init[:, 0:2],
        metric.gp.Z,
        metric.gp.kernel,
        mean_func=metric.gp.mean_func,
        f=metric.gp.q_mu,
        full_cov=True,
        q_sqrt=metric.gp.q_sqrt,
        cov_weight=metric.cov_weight,
    )
    metric_tensor_opt, _, _ = gp_metric_tensor(
        traj_opt[:, 0:2],
        metric.gp.Z,
        metric.gp.kernel,
        mean_func=metric.gp.mean_func,
        f=metric.gp.q_mu,
        full_cov=True,
        q_sqrt=metric.gp.q_sqrt,
        cov_weight=metric.cov_weight,
    )

    metric_trace_init = np.trace(metric_tensor_init, axis1=1, axis2=2)
    metric_trace_opt = np.trace(metric_tensor_opt, axis1=1, axis2=2)

    plot_3d_traj_col(fig, ax, traj_init, zs=metric_trace_init, color="c")
    plot_3d_traj_col(fig, ax, traj_opt, zs=metric_trace_opt, color="m")
    return fig, ax


def plot_3d_traj_col(fig, ax, traj, zs, color="k"):
    # ax.scatter(traj[:, 0], traj[:, 1], zs=zs, marker='x', color='k')
    # ax.plot3D(traj[:, 0], traj[:, 1], zs=zs.flatten(), color='k')
    ax.scatter(traj[:, 0], traj[:, 1], zs=0, zdir="z", color=color, marker="x")
    ax.scatter(traj[:, 0], 3, zs=zs, zdir="z", color=color, marker="x")
    ax.scatter(-2, traj[:, 1], zs=zs, zdir="z", color=color, marker="x")
    ax.plot3D(
        traj[:, 0],
        traj[:, 1],
        zs=0 * np.ones([*traj[:, 0].shape]),
        zdir="z",
        color=color,
    )
    ax.plot3D(
        traj[:, 0],
        3 * np.ones([*traj[:, 0].shape]),
        zs=zs.flatten(),
        zdir="z",
        color=color,
    )
    ax.plot3D(
        -2 * np.ones([*traj[:, 0].shape]),
        traj[:, 1],
        zs=zs.flatten(),
        zdir="z",
        color=color,
    )
    return fig, ax


def plot_3d_traj(fig, ax, traj, zs):
    # ax.scatter(traj[:, 0], traj[:, 1], zs=zs, marker='x', color='k')
    # ax.plot3D(traj[:, 0], traj[:, 1], zs=zs.flatten(), color='k')
    ax.scatter(traj[:, 0], traj[:, 1], zs=0, zdir="z", color="k", marker="x")
    ax.scatter(traj[:, 0], 3, zs=zs, zdir="z", color="c", marker="x")
    ax.scatter(-2, traj[:, 1], zs=zs, zdir="z", color="m", marker="x")
    ax.plot3D(
        traj[:, 0],
        traj[:, 1],
        zs=0 * np.ones([*traj[:, 0].shape]),
        zdir="z",
        color="k",
    )
    ax.plot3D(
        traj[:, 0],
        3 * np.ones([*traj[:, 0].shape]),
        zs=zs.flatten(),
        zdir="z",
        color="c",
    )
    ax.plot3D(
        -2 * np.ones([*traj[:, 0].shape]),
        traj[:, 1],
        zs=zs.flatten(),
        zdir="z",
        color="m",
    )
    return fig, ax
