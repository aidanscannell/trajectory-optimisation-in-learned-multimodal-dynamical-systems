import jax
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from gpjax.prediction import gp_jacobian, gp_predict
from jax.config import config
from tromp.metric_tensor import gp_metric_tensor
from tromp.mogpe import single_mogpe_mixing_probability
from tromp.visualisation.gp import plot_contourf, plot_mean_and_var
from tromp.visualisation.utils import create_grid

config.update("jax_enable_x64", True)

from jax import numpy as np

params = {
    "axes.labelsize": 15,
    "font.size": 15,
    # 'legend.fontsize': 15, # used for mixing prob
    "legend.fontsize": 10,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "text.usetex": True,
    # 'text.latex.preamble': ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
}
params = {
    "axes.labelsize": 30,
    "font.size": 25,
    # 'legend.fontsize': 15, # used for mixing prob
    "legend.fontsize": 15,
    # 'xtick.labelsize': 20, # used of mixing prob
    # 'ytick.labelsize': 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "text.usetex": True,
    # 'text.latex.preamble': ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
}
plt.rcParams.update(params)
color_init = "c"
color_opt = "m"
color_opt_1 = "m"
color_opt_2 = "olive"
# color_opts = [color_opt_1, color_opt_2]
color_opts = [color_opt_2, color_opt_1]


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
        # label='Init traj')
        label="Init traj",
    )
    ax.plot(solver.times, metric_trace_opt, color=color_opt, label="Opt traj")
    # label='Optimised trajectory')
    ax.legend()


def plot_mixing_prob_over_time(gp, solver, traj_init, traj_opts, labels):
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
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\Pr(\\alpha_t=1 | \mathbf{x}_t)$")

    ax.plot(
        solver.times,
        mixing_probs_init[:, 0:1],
        color=color_init,
        label="Init traj",
    )
    print("mixing probs init")
    print(mixing_probs_init.shape)
    sum_mixing_probs = np.sum(mixing_probs_init)
    print("sum mixing probs: ", sum_mixing_probs)

    for traj_opt, label, color in zip(traj_opts, labels, color_opts):
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
        print("mixing probs")
        print(mixing_probs_opt.shape)
        sum_mixing_probs = np.sum(mixing_probs_opt)
        print("sum mixing probs: ", sum_mixing_probs)
        ax.plot(
            solver.times, mixing_probs_opt[:, 0:1], color=color, label=label
        )
    save_name = dir_name + "mixing_prob_vs_time_no_legent.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)
    ax.legend()


def plot_start_and_end_pos(fig, ax, solver):
    fan_x = 2.5
    fan_y = -0.1
    fan_delta_x = 0.5
    fan_delta_y = 0.5
    ax.annotate(
        "Fan",
        (fan_x - 0.2, fan_y + 0.1),
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.plot(
        [fan_x, fan_x - fan_delta_x],
        [fan_y, fan_y + fan_delta_y],
        color="k",
        linewidth=4,
    )
    ax.plot(
        [fan_x, fan_x - fan_delta_x],
        [fan_y, fan_y - fan_delta_y],
        color="k",
        linewidth=4,
    )
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


def plot_svgp_and_all_trajs(gp, solver, traj_opts=None, labels=None):
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
        llabel="$\mathbb{E}[h^{(1)}]$",
        rlabel="$\mathbb{V}[h^{(1)}]$",
    )

    for ax in axs:
        fig, ax = plot_start_and_end_pos(fig, ax, solver)
        plot_omitted_data(fig, ax, color="k")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        # ax.scatter(gp.X[:, 0], gp.X[:, 1])
        plot_traj(
            fig, ax, solver.state_guesses, color=color_init, label="Init traj"
        )
        save_name = dir_name + "svgp_2d_traj_init.pdf"
        plt.savefig(
            save_name, transparent=True, bbox_inches="tight", pad_inches=0
        )

    if traj_opts is not None:
        i = 0
        for traj, label, color_opt in zip(traj_opts, labels, color_opts):
            for ax in axs:
                plot_traj(fig, ax, traj, color=color_opt, label=label)
            save_name = dir_name + "svgp_2d_traj_" + str(i) + ".pdf"
            plt.savefig(
                save_name, transparent=True, bbox_inches="tight", pad_inches=0
            )
            i += 1

    # axs[0].legend(loc='lower left')
    return fig, axs


def plot_svgp_and_start_end(gp, solver, traj_opts=None, labels=["", ""]):
    params = {
        "text.usetex": True,
        "text.latex.preamble": [
            "\\usepackage{amssymb}",
            "\\usepackage{amsmath}",
        ],
    }
    plt.rcParams.update(params)

    # Xnew, xx, yy = create_grid(gp.X, N=961)
    Xnew, xx, yy = create_grid(gp.Z, N=961)
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
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        # ax.scatter(gp.X[:, 0], gp.X[:, 1])
        plot_traj(
            fig, ax, solver.state_guesses, color=color_init, label="Init traj"
        )
        if traj_opts is not None:
            if isinstance(traj_opts, list):
                for traj, label, color_opt in zip(
                    traj_opts, labels, color_opts
                ):
                    plot_traj(fig, ax, traj, color=color_opt, label=label)
            else:
                plot_traj(fig, ax, traj_opts)
    axs[0].legend(loc="lower left")
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
            fig, ax, solver.state_guesses, color=color_init, label="Init traj"
        )
        if traj_opt is not None:
            plot_traj(fig, ax, traj_opt, color=color_opt, label="Opt traj")
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


def plot_epistemic_var_vs_time(
    gp, solver, traj_init, traj_opts=None, labels=None
):
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
    sum_var_init = np.sum(var_init)
    print("Sum epistemic var init = ", sum_var_init)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\mathbb{V}[h^{(1)}]$")

    ax.plot(solver.times, var_init[0, :], color=color_init, label="Init traj")

    var_opt = 0.0
    for traj_opt, label, color in zip(traj_opts, labels, color_opts):
        _, var_opt = gp_predict(
            traj_opt[:, 0:2],
            gp.Z,
            kernels=gp.kernel,
            mean_funcs=gp.mean_func,
            f=gp.q_mu,
            q_sqrt=gp.q_sqrt,
            full_cov=False,
        )
        ax.plot(solver.times, var_opt[0, :], color=color, label=label)
        sum_var_opt = np.sum(var_opt)
        var_opt = 0.0
        print("Sum epistemic var opt = ", sum_var_opt)

    ax.legend()
    return fig, ax


def plot_aleatoric_var_vs_time(
    gp, solver, traj_init, traj_opts=None, labels=None
):
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

    noise_vars = np.array(gp.noise_vars).reshape(-1, 1)
    var_init = mixing_probs_init @ noise_vars
    sum_var_init = np.sum(var_init)
    print("Sum aleatoric var init = ", sum_var_init)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\sum_{k=1}^K\Pr(\\alpha=k|\mathbf{x}) (\sigma^{(k)})^2$")

    ax.plot(solver.times, var_init, color=color_init, label="Init traj")

    for traj_opt, color, label in zip(traj_opts, color_opts, labels):
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

        var_opt = mixing_probs_opt @ noise_vars

        ax.plot(solver.times, var_opt, color=color, label=label)
        sum_var_opt = np.sum(var_opt)
        print("Sum aleatoric var opt = ", sum_var_opt)

    ax.legend()
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


def plot_mixing_prob_all_trajs(gp, solver, traj_opts=None, labels=None):

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)

    mixing_probs = jax.vmap(
        single_mogpe_mixing_probability,
        (0, None, None, None, None, None, None),
    )(Xnew, gp.Z, gp.kernel, gp.mean_func, gp.q_mu, False, gp.q_sqrt)
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

    # plot_omitted_data(fig, ax, color='k')
    plot_start_and_end_pos(fig, ax, solver)

    plot_traj(
        fig, ax, solver.state_guesses, color=color_init, label="Init traj"
    )

    save_name = dir_name + "mixing_prob_2d_traj_init.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    if traj_opts is not None:
        i = 0
        for traj, label, color in zip(traj_opts, labels, color_opts):
            plot_traj(fig, ax, traj, color=color, label=label)
            save_name = dir_name + "mixing_prob_2d_traj_" + str(i) + ".pdf"
            plt.savefig(
                save_name, transparent=True, bbox_inches="tight", pad_inches=0
            )
            i += 1

    ax.legend(loc=3)

    return fig, ax


def plot_mixing_prob_and_start_end(gp, solver, traj_opts=None, labels=None):

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)

    mixing_probs = jax.vmap(
        single_mogpe_mixing_probability,
        (0, None, None, None, None, None, None),
    )(Xnew, gp.Z, gp.kernel, gp.mean_func, gp.q_mu, False, gp.q_sqrt)
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
        fig, ax, solver.state_guesses, color=color_init, label="Init traj"
    )
    if traj_opts is not None:
        for traj, label, color in zip(traj_opts, labels, color_opts):
            plot_traj(fig, ax, traj, color=color, label=label)
    ax.legend(loc=3)

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


def plot_svgp_metric_trace_and_start_end(
    metrics, solver, traj_opts, labels, linestyles
):

    # plot original GP
    Xnew, xx, yy = create_grid(metrics[0].gp.X, N=961)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Tr$(\mathbf{G}(\mathbf{x}(t)))$")

    for traj_opt, metric, color, label, linetyle in zip(
        traj_opts, metrics, color_opts, labels, linestyles
    ):
        metric_tensor_init, _, _ = gp_metric_tensor(
            solver.state_guesses[:, 0:2],
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

        traces = np.stack([metric_trace_opt, metric_trace_init])
        max_trace = np.max(traces)
        min_trace = np.min(traces)
        print("max min")
        print(max_trace)
        print(min_trace)
        ax.plot(
            solver.times,
            (metric_trace_init - traces.min()) / (traces.max() - traces.min()),
            color=color_init,
            linestyle=linetyle,
            label="Init traj $\lambda=$" + label,
        )
        ax.plot(
            solver.times,
            (metric_trace_opt - traces.min()) / (traces.max() - traces.min()),
            color=color,
            linestyle=linetyle,
            label="Opt traj $\lambda=$" + label,
        )

    ax.legend()
    return fig, ax


def create_save_dir():
    import pathlib

    dir_name = "../../reports/figures/"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    return dir_name


if __name__ == "__main__":
    traj_1_filename = (
        "../../reports/figures/26-10-2020/144218/geodesic_traj.npz"
    )
    traj_2_filename = (
        "../../reports/figures/26-10-2020/143053/geodesic_traj.npz"
    )

    gp = FakeSVGPQuad()
    metric_low = FakeSVGPMetricQuadLow()
    metric_high = FakeSVGPMetricQuadHigh()
    metrics = [metric_low, metric_high]
    solver = FakeCollocationSVGPQuad()
    params = {
        "text.usetex": True,
    }
    plt.rcParams.update(params)

    dir_name = create_save_dir()
    geodesic_traj_1 = np.load(traj_1_filename)["arr_0"]
    geodesic_traj_2 = np.load(traj_2_filename)["arr_0"]
    print("geodesic")
    print(geodesic_traj_2)

    traj_opts = [geodesic_traj_2, geodesic_traj_1]
    labels = ["Opt traj $\lambda=0.5$", "Opt traj $\lambda=20$"]
    labels_metric = ["0.5", "20"]
    linestyles = ["--", "-"]
    # traj_opts = [geodesic_traj_1, geodesic_traj_2]
    # labels = ['Opt traj $\lambda=20$', 'Opt traj $\lambda=0.5$']
    # labels_metric = ['20', '0.5']
    # linestyles = ['-', '--']

    plot_mixing_prob_over_time(
        gp,
        solver,
        traj_init=solver.state_guesses,
        traj_opts=traj_opts,
        labels=labels,
    )
    save_name = dir_name + "mixing_prob_over_time.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    fig, axs = plot_epistemic_var_vs_time(
        gp,
        solver,
        traj_init=solver.state_guesses,
        traj_opts=traj_opts,
        labels=labels,
    )
    save_name = dir_name + "epistemic_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.show()

    plot_svgp_and_all_trajs(gp, solver, traj_opts=traj_opts, labels=labels)
    plot_mixing_prob_all_trajs(gp, solver, traj_opts=traj_opts, labels=labels)
    plt.show()

    # plot metric traces over time
    fig, axs = plot_svgp_metric_trace_and_start_end(
        metrics,
        solver,
        traj_opts=traj_opts,
        labels=labels_metric,
        linestyles=linestyles,
    )
    save_name = dir_name + "metric_trace_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    # plot init and opt trajectories over svgp
    fig, axs = plot_svgp_and_start_end(
        gp, solver, traj_opts=traj_opts, labels=labels
    )
    save_name = dir_name + "svgp_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    # init and opt trajectories over mixing probability
    fig, ax = plot_mixing_prob_and_start_end(
        gp, solver, traj_opts=traj_opts, labels=labels
    )
    save_name = dir_name + "mixing_prob_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.show()

    # # # init and opt trajectories over metric trace
    # fig, axs = plot_svgp_metric_trace_and_start_end(metrics,
    #                                                 solver,
    #                                                 traj_opts=traj_opts,
    #                                                 labels=labels_metric)
    # save_name = dir_name + "metric_trace_2d_traj.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # prob vs time
    plot_mixing_prob_over_time(
        gp,
        solver,
        traj_init=solver.state_guesses,
        traj_opts=traj_opts,
        labels=labels,
    )
    save_name = dir_name + "mixing_prob_vs_time.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    # # plot metric trace vs time
    # plot_metirc_trace_over_time(metric,
    #                             solver,
    #                             traj_init=solver.state_guesses,
    #                             traj_opt=geodesic_traj)
    # save_name = dir_name + "metric_trace_vs_time.pdf"
    # plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # plot epistemic uncertainty vs time
    fig, axs = plot_epistemic_var_vs_time(
        gp,
        solver,
        traj_init=solver.state_guesses,
        traj_opts=traj_opts,
        labels=labels,
    )
    save_name = dir_name + "epistemic_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    plot_aleatoric_var_vs_time(
        gp,
        solver,
        traj_init=solver.state_guesses,
        traj_opts=traj_opts,
        labels=labels,
    )
    save_name = dir_name + "aleatoric_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches="tight", pad_inches=0)

    plt.show()
