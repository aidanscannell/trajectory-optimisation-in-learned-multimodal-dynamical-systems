import jax
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from gpjax.prediction import gp_jacobian, gp_predict
from jax import numpy as np
from jax.config import config
from ProbGeo.metric_tensor import gp_metric_tensor
from ProbGeo.mogpe import (mogpe_mixing_probability,
                           single_mogpe_mixing_probability)
from ProbGeo.visualisation.gp import plot_contourf, plot_mean_and_var
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


def plot_metirc_trace_over_time(metric, solver, traj_init, traj_opt):
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
    metric_trace_init = np.trace(metric_tensor_init, axis1=1, axis2=2)
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
    metric_trace_opt = np.trace(metric_tensor_opt, axis1=1, axis2=2)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Tr$(G(\mathbf{x}_t))$")

    ax.plot(
        solver.times,
        metric_trace_init,
        color=color_init,
        label="Initial trajectory",
    )
    ax.plot(
        solver.times,
        metric_trace_opt,
        color=color_opt,
        label="Optimised trajectory",
    )
    ax.legend()


def plot_mixing_prob_over_time(gp, solver, traj_init, traj_opt):
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
    # rcParams["figure.figsize"] = [6.4, 2.8]
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\Pr(\\alpha_t=1 | \mathbf{x}_t)$")

    ax.plot(
        solver.times,
        mixing_probs_init[:, 0:1],
        color=color_init,
        label="Initial trajectory",
    )
    ax.plot(
        solver.times,
        mixing_probs_opt[:, 0:1],
        color=color_opt,
        label="Optimised trajectory",
    )
    ax.legend()


def plot_gp_and_start_end(gp, solver):
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(
        Xnew,
        gp.X,
        kernels=gp.kernel,
        mean_funcs=gp.mean_func,
        f=gp.Y,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    for ax in axs:
        fig, ax = plot_start_and_end_pos(fig, ax, solver)
    return fig, axs


def plot_start_and_end_pos(fig, ax, solver):
    ax.scatter(solver.pos_init[0], solver.pos_init[1], marker="x", color="k")
    ax.scatter(
        solver.pos_end_targ[0], solver.pos_end_targ[1], color="k", marker="x"
    )
    ax.annotate(
        "Start $\mathbf{x}_0$",
        (solver.pos_init[0], solver.pos_init[1]),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.annotate(
        "End $\mathbf{x}_f$",
        (solver.pos_end_targ[0], solver.pos_end_targ[1]),
        horizontalalignment="left",
        verticalalignment="top",
    )
    return fig, ax


def plot_svgp_and_start_end(gp, solver, traj_opt=None):
    params = {
        "text.usetex": True,
        "text.latex.preamble": [
            "\\usepackage{amssymb}",
            "\\usepackage{amsmath}",
        ],
    }
    plt.rcParams.update(params)

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
    print("mu var")
    # print(mu.shape)
    # print(var.shape)
    # mu = mu[0:1, :, :]
    # var = var[0:1, :]
    # mu = mu[1:2, :, :]
    # var = var[1:2, :]
    print(mu.shape)
    print(var.shape)
    fig, axs = plot_mean_and_var(
        xx,
        yy,
        mu,
        var,
        llabel="$\mathbb{E}[h^{(1)}(\mathbf{x})]$",
        rlabel="$\mathbb{V}[h^{(1)}(\mathbf{x})]$",
    )

    for ax in axs:
        fig, ax = plot_start_and_end_pos(fig, ax, solver)
        # plot_omitted_data(fig, ax, color="k")
        # ax.scatter(gp.X[:, 0], gp.X[:, 1])
        plot_traj(
            fig,
            ax,
            solver.state_guesses,
            color=color_init,
            label="Initial trajectory",
        )
        if traj_opt is not None:
            plot_traj(
                fig,
                ax,
                traj_opt,
                color=color_opt,
                label="Optimised trajectory",
            )
    axs[0].legend()
    return fig, axs


def plot_svgp_jacobian_mean(gp, solver, traj_opt=None):
    params = {
        "text.usetex": True,
        "text.latex.preamble": [
            "\\usepackage{amssymb}",
            "\\usepackage{amsmath}",
        ],
    }
    plt.rcParams.update(params)

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

    def gp_jacobian_all(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return gp_jacobian(
            x,
            gp.Z,
            gp.kernel,
            gp.mean_func,
            f=gp.q_mu,
            q_sqrt=gp.q_sqrt,
            full_cov=False,
        )

    mu_j, var_j = jax.vmap(gp_jacobian_all, in_axes=(0))(Xnew)
    print("gp jacobain mu var")
    print(mu_j.shape)
    print(var_j.shape)
    # mu = np.prod(mu, 1)
    # var = np.diagonal(var, axis1=-2, axis2=-1)
    # var = np.prod(var, 1)
    fig, axs = plot_mean_and_var(
        xx,
        yy,
        mu,
        var,
        # mu,
        # var,
        llabel="$\mathbb{E}[h^{(1)}]$",
        rlabel="$\mathbb{V}[h^{(1)}]$",
    )

    for ax in axs:
        ax.quiver(Xnew[:, 0], Xnew[:, 1], mu_j[:, 0], mu_j[:, 1], color="k")

        fig, ax = plot_start_and_end_pos(fig, ax, solver)
        plot_omitted_data(fig, ax, color="k")
        # ax.scatter(gp.X[:, 0], gp.X[:, 1])
        plot_traj(
            fig,
            ax,
            solver.state_guesses,
            color=color_init,
            label="Initial trajectory",
        )
        if traj_opt is not None:
            plot_traj(
                fig,
                ax,
                traj_opt,
                color=color_opt,
                label="Optimised trajectory",
            )
    axs[0].legend()
    return fig, axs


def plot_svgp_jacobian_var(gp, solver, traj_opt=None):
    params = {
        "text.usetex": True,
        "text.latex.preamble": [
            "\\usepackage{amssymb}",
            "\\usepackage{amsmath}",
        ],
    }
    plt.rcParams.update(params)
    input_dim = gp.X.shape[1]

    Xnew, xx, yy = create_grid(gp.X, N=961)

    def gp_jacobian_all(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return gp_jacobian(
            x,
            gp.Z,
            gp.kernel,
            gp.mean_func,
            f=gp.q_mu,
            q_sqrt=gp.q_sqrt,
            full_cov=False,
        )

    mu_j, var_j = jax.vmap(gp_jacobian_all, in_axes=(0))(Xnew)
    print("gp jacobain mu var")
    print(mu_j.shape)
    print(var_j.shape)
    # mu = np.prod(mu, 1)
    # var = np.diagonal(var, axis1=-2, axis2=-1)
    # var = np.prod(var, 1)

    fig, axs = plt.subplots(input_dim, input_dim)
    for i in range(input_dim):
        for j in range(input_dim):
            plot_contourf(
                fig,
                axs[i, j],
                xx,
                yy,
                var_j[:, i, j],
                label="$\Sigma_J(\mathbf{x})$",
            )
            axs[i, j].set_xlabel("$x$")
            axs[i, j].set_ylabel("$y$")

    return fig, axs


def plot_svgp_metric_and_start_end(metric, solver, traj_opt=None):
    params = {
        "text.usetex": True,
        "text.latex.preamble": [
            "\\usepackage{amssymb}",
            "\\usepackage{amsmath}",
        ],
    }
    plt.rcParams.update(params)

    input_dim = metric.gp.X.shape[1]
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
    print("metric yo yo")
    print(metric_tensor.shape)

    fig, axs = plt.subplots(input_dim, input_dim)
    for i in range(input_dim):
        for j in range(input_dim):
            plot_contourf(
                fig,
                axs[i, j],
                xx,
                yy,
                metric_tensor[:, i, j],
                label="$G(\mathbf{x})$",
            )
            axs[i, j].set_xlabel("$x$")
            axs[i, j].set_ylabel("$y$")

    return fig, axs


def plot_epistemic_var_vs_time(gp, solver, traj_init, traj_opt=None):
    params = {
        "text.usetex": True,
        "text.latex.preamble": [
            "\\usepackage{amssymb}",
            "\\usepackage{amsmath}",
        ],
    }
    plt.rcParams.update(params)

    _, var_init = gp_predict(
        traj_init[:, 0:2],
        gp.Z,
        kernels=gp.kernel,
        mean_funcs=gp.mean_func,
        f=gp.q_mu,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    var_opt = 0.0
    if traj_opt is not None:
        _, var_opt = gp_predict(
            traj_opt[:, 0:2],
            gp.Z,
            kernels=gp.kernel,
            mean_funcs=gp.mean_func,
            f=gp.q_mu,
            q_sqrt=gp.q_sqrt,
            full_cov=False,
        )
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\mathbb{V}[h^{(1)}]$")

    ax.plot(
        solver.times,
        var_init[0, :],
        color=color_init,
        label="Initial trajectory",
    )
    if traj_opt is not None:
        ax.plot(
            solver.times,
            var_opt[0, :],
            color=color_opt,
            label="Optimised trajectory",
        )
    ax.legend()
    sum_var_init = np.sum(var_init)
    sum_var_opt = np.sum(var_opt)
    print("Sum epistemic var init = ", sum_var_init)
    print("Sum epistemic var opt = ", sum_var_opt)
    return fig, ax


def plot_aleatoric_var_vs_time(gp, solver, traj_init, traj_opt=None):
    params = {
        "text.usetex": True,
        "text.latex.preamble": [
            "\\usepackage{amssymb}",
            "\\usepackage{amsmath}",
        ],
    }
    plt.rcParams.update(params)

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
    if mixing_probs_init.shape[-1] == 1:
        mixing_probs_init = np.concatenate(
            [mixing_probs_init, 1 - mixing_probs_init], -1
        )
    if traj_opt is not None:
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
        if mixing_probs_opt.shape[-1] == 1:
            mixing_probs_opt = np.concatenate(
                [mixing_probs_opt, 1 - mixing_probs_opt], -1
            )

    noise_vars = np.array(gp.noise_vars).reshape(-1, 1)
    var_init = mixing_probs_init @ noise_vars

    var_opt = 0
    if traj_opt is not None:
        var_opt = mixing_probs_opt @ noise_vars
        print("var opt")
        print(var_opt.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\sum_{k=1}^K\Pr(\\alpha=k|\mathbf{x}) (\sigma^{(k)})^2$")

    ax.plot(
        solver.times, var_init, color=color_init, label="Initial trajectory"
    )
    if traj_opt is not None:
        ax.plot(
            solver.times,
            var_opt,
            color=color_opt,
            label="Optimised trajectory",
        )
    ax.legend()
    sum_var_init = np.sum(var_init)
    sum_var_opt = np.sum(var_opt)
    print("Sum aleatoric var init = ", sum_var_init)
    print("Sum aleatoric var opt = ", sum_var_opt)
    return fig, ax


def plot_traj(fig, axs, traj, color="k", label=""):
    try:
        for ax in axs:
            ax.scatter(traj[:, 0], traj[:, 1], marker="x", color=color)
            ax.plot(
                traj[:, 0], traj[:, 1], marker="x", color=color, label=label
            )
    except:
        axs.scatter(traj[:, 0], traj[:, 1], marker="x", color=color)
        axs.plot(traj[:, 0], traj[:, 1], marker="x", color=color, label=label)
    return fig, axs


def plot_mixing_prob_and_start_end(gp, solver, traj_opt=None):

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)

    # TODO need to change gp.X to gp.Z (and gp.q_mu) for sparse
    mixing_probs = jax.vmap(
        single_mogpe_mixing_probability,
        (0, None, None, None, None, None, None),
    )(
        Xnew,
        # gp.X,
        gp.Z,
        gp.kernel,
        gp.mean_func,
        gp.q_mu,
        False,
        gp.q_sqrt,
    )
    # mixing_probs = mogpe_mixing_probability(Xnew,
    #                                         gp.X,
    #                                         gp.kernel,
    #                                         mean_func=gp.mean_func,
    #                                         f=gp.Y,
    #                                         q_sqrt=gp.q_sqrt,
    #                                         full_cov=False)
    # print('mixing probs yo')
    # print(mixing_probs.shape)
    # mixing_probs = mixing_probs[:, 0, :] * mixing_probs[:, 1, :]
    # output_dim = mixing_probs.shape[0]
    fig, ax = plt.subplots(1, 1)
    plot_contourf(
        fig,
        ax,
        xx,
        yy,
        # mixing_probs[:, 1:2],
        mixing_probs[:, 0:1],
        label="$\Pr(\\alpha=1 | \mathbf{x})$",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    plot_omitted_data(fig, ax, color="k")

    plot_start_and_end_pos(fig, ax, solver)

    plot_traj(
        fig,
        ax,
        solver.state_guesses,
        color=color_init,
        label="Initial trajectory",
    )
    if traj_opt is not None:
        plot_traj(
            fig, ax, traj_opt, color=color_opt, label="Optimised trajectory"
        )
    ax.legend()

    return fig, ax


def plot_omitted_data(
    fig, ax, x1_low=-0.5, x2_low=-3.0, x1_high=1.0, x2_high=1.0, color="r"
):
    x1_low = -1.0
    x2_low = -1.0
    x1_high = 1.0
    x2_high = 2.5
    ax.add_patch(
        patches.Rectangle(
            (x1_low, x2_low),
            x1_high - x1_low,
            x2_high - x2_low,
            edgecolor=color,
            facecolor="red",
            hatch="/",
            label="No observations",
            fill=False,
        )
    )

    return fig, ax


def plot_svgp_mixing_prob_and_start_end(gp, solver, traj_opt=None):

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mixing_probs = mogpe_mixing_probability(
        Xnew,
        gp.Z,
        gp.kernel,
        mean_func=gp.mean_func,
        f=gp.q_mu,
        q_sqrt=gp.q_sqrt,
        full_cov=False,
    )
    fig, ax = plt.subplots(1, 1)
    plot_contourf(fig, ax, xx, yy, mixing_probs[:, 0:1])

    plot_start_and_end_pos(fig, ax, solver)
    plot_traj(
        fig,
        ax,
        solver.state_guesses,
        color=color_init,
        label="Initial trajectory",
    )
    if traj_opt is not None:
        plot_traj(
            fig, ax, traj_opt, color=color_opt, label="Optimised trajectory"
        )
    ax.legend()
    return fig, ax


def plot_svgp_metric_trace_and_start_end(metric, solver, traj_opt=None):

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
    fig, ax = plt.subplots(1, 1)
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    surf_traceG = plot_contourf(
        fig, ax, xx, yy, metric_trace, label="Tr$(G(\mathbf{x}))$"
    )
    plot_start_and_end_pos(fig, ax, solver)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plot_traj(
        fig,
        ax,
        solver.state_guesses,
        color=color_init,
        label="Initial trajectory",
    )
    if traj_opt is not None:
        plot_traj(
            fig, ax, traj_opt, color=color_opt, label="Optimised trajectory"
        )
    ax.legend()
    return fig, ax


def test_and_plot_collocation_svgp_quad():
    gp = FakeSVGPQuad()
    metric = FakeSVGPMetricQuad()
    solver = FakeCollocationSVGPQuad()
    params = {
        # 'axes.labelsize': 30,
        # 'font.size': 30,
        # 'legend.fontsize': 20,
        # 'xtick.labelsize': 30,
        # 'ytick.labelsize': 30,
        "text.usetex": True,
    }
    plt.rcParams.update(params)

    dir_name = create_save_dir()
    fig, ax = plot_mixing_prob_and_start_end(gp, solver, solver.state_guesses)
    # plot_traj(fig, ax, solver.state_guesses)
    save_name = dir_name + "mixing_prob_2d_init_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    fig, axs = plot_svgp_and_start_end(gp, solver)
    # plot_traj(fig, axs, solver.state_guesses)
    save_name = dir_name + "svgp_2d_init_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    fig, axs = plot_epistemic_var_vs_time(
        gp, solver, traj_init=solver.state_guesses, traj_opt=None
    )
    save_name = dir_name + "epistemic_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)
    plot_aleatoric_var_vs_time(
        gp, solver, traj_init=solver.state_guesses, traj_opt=None
    )
    save_name = dir_name + "aleatoric_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    fig, ax = plot_svgp_metric_trace_and_start_end(metric, solver)
    plot_traj(fig, ax, solver.state_guesses)
    save_name = dir_name + "metric_trace_2d_init_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    fig, axs = plot_svgp_jacobian_mean(
        gp, solver, traj_opt=solver.state_guesses
    )
    fig, axs = plot_svgp_jacobian_var(
        gp, solver, traj_opt=solver.state_guesses
    )

    fig, axs = plot_svgp_metric_and_start_end(
        metric, solver, traj_opt=solver.state_guesses
    )

    # # prob vs time
    # plot_mixing_prob_over_time(gp,
    #                            solver,
    #                            traj_init=solver.state_guesses,
    #                            traj_opt=solver.state_guesses)
    # save_name = dir_name + "mixing_prob_vs_time.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    # plot_metirc_trace_over_time(metric,
    #                             solver,
    #                             traj_init=solver.state_guesses,
    #                             traj_opt=solver.state_guesses)
    # save_name = dir_name + "metric_trace_vs_time.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # fig, axs = plot_3d_mean_and_var(gp, solver)
    # plot_3d_traj_mean_and_var(fig, axs, gp, traj=solver.state_guesses)
    # save_name = dir_name + "init_traj_mean_and_var.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    # fig, ax = plot_3d_metric_trace(metric, solver)
    # plot_3d_traj_metric_trace(fig, ax, metric, traj=solver.state_guesses)
    # save_name = dir_name + "init_traj_metric_trace.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    # fig, ax = plot_3d_mixing_prob(gp, solver)
    # plot_3d_traj_mixing_prob(fig, ax, gp, traj=solver.state_guesses)
    # save_name = dir_name + "init_traj_mixing_prob.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    plt.show()
    geodesic_traj = test_collocation_svgp_quad()

    # fig, ax = plot_3d_metric_trace(metric, solver)
    # plot_3d_traj_metric_trace(fig, ax, metric, traj=geodesic_traj)
    # save_name = dir_name + "opt_traj_metric_trace.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    # fig, axs = plot_3d_mean_and_var(gp, solver)
    # plot_3d_traj_mean_and_var(fig, axs, gp, geodesic_traj)
    # save_name = dir_name + "opt_traj_mean_and_var.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    # fig, ax = plot_3d_mixing_prob(gp, solver)
    # plot_3d_traj_mixing_prob(fig, ax, gp, traj=geodesic_traj)
    # save_name = dir_name + "opt_traj_mixing_prob.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # fig, axs = plot_svgp_metric_trace_and_start_end(metric, solver)
    # fig, axs = plot_traj(fig, axs, geodesic_traj)
    # fig, axs = plot_svgp_and_start_end(gp, solver)
    # fig, axs = plot_traj(fig, axs, geodesic_traj)
    # plt.show()

    # plot init and opt trajectories over svgp
    fig, axs = plot_svgp_and_start_end(gp, solver, traj_opt=geodesic_traj)
    save_name = dir_name + "svgp_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    # init and opt trajectories over mixing probability
    fig, ax = plot_mixing_prob_and_start_end(
        gp, solver, traj_opt=geodesic_traj
    )
    save_name = dir_name + "mixing_prob_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    # init and opt trajectories over metric trace
    fig, ax = plot_svgp_metric_trace_and_start_end(
        metric, solver, traj_opt=geodesic_traj
    )
    save_name = dir_name + "metric_trace_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    # prob vs time
    plot_mixing_prob_over_time(
        gp, solver, traj_init=solver.state_guesses, traj_opt=geodesic_traj
    )
    save_name = dir_name + "mixing_prob_vs_time.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)
    # plot metric trace vs time
    plot_metirc_trace_over_time(
        metric, solver, traj_init=solver.state_guesses, traj_opt=geodesic_traj
    )
    save_name = dir_name + "metric_trace_vs_time.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    # plot epistemic uncertainty vs time
    fig, axs = plot_epistemic_var_vs_time(
        gp, solver, traj_init=solver.state_guesses, traj_opt=geodesic_traj
    )
    save_name = dir_name + "epistemic_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)
    plot_aleatoric_var_vs_time(
        gp, solver, traj_init=solver.state_guesses, traj_opt=geodesic_traj
    )
    save_name = dir_name + "aleatoric_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    save_name = dir_name + "geodesic_traj.npz"
    np.savez(save_name, geodesic_traj)

    # fig, ax = plot_3d_mixing_prob(gp, solver)
    # plot_3d_traj_mixing_prob_init_and_opt(fig,
    #                                       ax,
    #                                       gp,
    #                                       traj_init=solver.state_guesses,
    #                                       traj_opt=geodesic_traj)
    # save_name = dir_name + "traj_mixing_prob.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # fig, ax = plot_3d_metric_trace(metric, solver)
    # plot_3d_traj_metric_trace_init_and_opt(fig,
    #                                        ax,
    #                                        metric,
    #                                        traj_init=solver.state_guesses,
    #                                        traj_opt=geodesic_traj)
    # save_name = dir_name + "traj_metric_trace.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    plt.show()
