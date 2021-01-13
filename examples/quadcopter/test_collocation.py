import jax
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from jax import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)

params = {
    # 'axes.labelsize': 30,
    # 'font.size': 30,
    # 'legend.fontsize': 20,
    # 'xtick.labelsize': 30,
    # 'ytick.labelsize': 30,
    'text.usetex': True,
    # 'text.latex.preamble': ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
}
plt.rcParams.update(params)
color_init = 'c'
color_opt = 'm'


def plot_metirc_trace_over_time(metric, solver, traj_init, traj_opt):
    from ProbGeo.mogpe import single_mogpe_mixing_probability
    from ProbGeo.metric_tensor import gp_metric_tensor
    metric_tensor_init, _, _ = gp_metric_tensor(traj_init[:, 0:2],
                                                metric.gp.Z,
                                                metric.gp.kernel,
                                                mean_func=metric.gp.mean_func,
                                                f=metric.gp.q_mu,
                                                full_cov=True,
                                                q_sqrt=metric.gp.q_sqrt,
                                                cov_weight=metric.cov_weight)
    metric_trace_init = np.trace(metric_tensor_init, axis1=1, axis2=2)
    metric_tensor_opt, _, _ = gp_metric_tensor(traj_opt[:, 0:2],
                                               metric.gp.Z,
                                               metric.gp.kernel,
                                               mean_func=metric.gp.mean_func,
                                               f=metric.gp.q_mu,
                                               full_cov=True,
                                               q_sqrt=metric.gp.q_sqrt,
                                               cov_weight=metric.cov_weight)
    metric_trace_opt = np.trace(metric_tensor_opt, axis1=1, axis2=2)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Tr$(G(\mathbf{x}_t))$')

    ax.plot(solver.times,
            metric_trace_init,
            color=color_init,
            label='Initial trajectory')
    ax.plot(solver.times,
            metric_trace_opt,
            color=color_opt,
            label='Optimised trajectory')
    ax.legend()


def plot_mixing_prob_over_time(gp, solver, traj_init, traj_opt):
    from ProbGeo.mogpe import single_mogpe_mixing_probability
    mixing_probs_init = jax.vmap(single_mogpe_mixing_probability,
                                 (0, None, None, None, None, None, None))(
                                     traj_init[:, 0:2], gp.Z, gp.kernel,
                                     gp.mean_func, gp.q_mu, False, gp.q_sqrt)
    mixing_probs_opt = jax.vmap(single_mogpe_mixing_probability,
                                (0, None, None, None, None, None, None))(
                                    traj_opt[:, 0:2], gp.Z, gp.kernel,
                                    gp.mean_func, gp.q_mu, False, gp.q_sqrt)
    # rcParams["figure.figsize"] = [6.4, 2.8]
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$\Pr(\\alpha_t=1 | \mathbf{x}_t)$')

    ax.plot(solver.times,
            mixing_probs_init[:, 0:1],
            color=color_init,
            label='Initial trajectory')
    ax.plot(solver.times,
            mixing_probs_opt[:, 0:1],
            color=color_opt,
            label='Optimised trajectory')
    ax.legend()


def plot_3d_surf(fig, ax, x, y, z):
    from matplotlib import cm
    surf = ax.plot_surface(x,
                           y,
                           z.reshape([*x.shape]),
                           rstride=1,
                           cstride=1,
                           alpha=0.7,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.elev = 30
    ax.azim = -41
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return fig, ax


def plot_3d_mean_and_var(gp, solver):
    import matplotlib.pyplot as plt
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict

    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(Xnew,
                         gp.Z,
                         kernels=gp.kernel,
                         mean_funcs=gp.mean_func,
                         f=gp.q_mu,
                         q_sqrt=gp.q_sqrt,
                         full_cov=False)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_mu = fig.add_subplot(1, 2, 1, projection="3d")
    surf_mu = plot_3d_surf(fig, ax_mu, xx, yy, mu)
    ax_var = fig.add_subplot(1, 2, 2, projection="3d")
    surf_var = plot_3d_surf(fig, ax_var, xx, yy, var)

    axs = [ax_mu, ax_var]
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(-40, 40)
    # ax.set_zlim(-100, 100)

    # ax_mu.set_xlabel('$x$')
    # ax_mu.set_ylabel('$y$')
    # ax_var.set_xlabel('$x$')
    # ax_var.set_ylabel('$y$')
    ax_mu.set_zlabel('Mean')
    ax_var.set_zlabel('Variance')

    # plt.show()
    return fig, axs


def plot_3d_metric_trace(metric, solver):
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.metric_tensor import gp_metric_tensor, metric_tensor_fn

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
        cov_weight=metric.cov_weight)

    metric_trace = np.trace(metric_tensor, axis1=1, axis2=2)
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surf = plot_3d_surf(fig, ax, xx, yy, metric_trace)
    ax.set_zlabel('Tr$(G(\mathbf{x}_*))$')

    return fig, ax


def plot_3d_mixing_prob(gp, solver):
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.mogpe import single_mogpe_mixing_probability

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)

    mixing_probs = jax.vmap(single_mogpe_mixing_probability,
                            (0, None, None, None, None, None, None))(
                                Xnew, gp.Z, gp.kernel, gp.mean_func, gp.q_mu,
                                False, gp.q_sqrt)

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surf = plot_3d_surf(fig, ax, xx, yy, mixing_probs[:, 0:1])
    ax.set_zlabel('$\Pr(\\alpha_*=0 | \mathbf{x}_*)$')

    return fig, ax


def plot_3d_traj_mean_and_var(fig, axs, gp, traj):
    from ProbGeo.gp import gp_predict
    from ProbGeo.visualisation.utils import create_grid
    Xnew, xx, yy = create_grid(gp.X, N=961)
    traj_mu, traj_var = gp_predict(traj[:, 0:2],
                                   gp.Z,
                                   kernels=gp.kernel,
                                   mean_funcs=gp.mean_func,
                                   f=gp.q_mu,
                                   q_sqrt=gp.q_sqrt,
                                   full_cov=False)
    plot_3d_traj(fig, axs[0], traj, zs=traj_mu)
    plot_3d_traj(fig, axs[1], traj, zs=traj_var)
    return fig, axs


def plot_3d_traj_metric_trace(fig, ax, metric, traj):
    from ProbGeo.metric_tensor import gp_metric_tensor, metric_tensor_fn
    metric_tensor, mu_j, cov_j = gp_metric_tensor(
        traj[:, 0:2],
        metric.gp.Z,
        metric.gp.kernel,
        mean_func=metric.gp.mean_func,
        f=metric.gp.q_mu,
        full_cov=True,
        q_sqrt=metric.gp.q_sqrt,
        cov_weight=metric.cov_weight)

    metric_trace = np.trace(metric_tensor, axis1=1, axis2=2)

    plot_3d_traj(fig, ax, traj, zs=metric_trace)
    return fig, ax


def plot_3d_traj_mixing_prob(fig, ax, gp, traj):
    from ProbGeo.mogpe import single_mogpe_mixing_probability

    mixing_probs = jax.vmap(single_mogpe_mixing_probability,
                            (0, None, None, None, None, None, None))(
                                traj[:, 0:2], gp.Z, gp.kernel, gp.mean_func,
                                gp.q_mu, False, gp.q_sqrt)

    plot_3d_traj(fig, ax, traj, zs=mixing_probs[:, 0:1])
    return fig, ax


def plot_3d_traj_mixing_prob_init_and_opt(fig, ax, gp, traj_init, traj_opt):
    from ProbGeo.mogpe import single_mogpe_mixing_probability

    mixing_probs_init = jax.vmap(single_mogpe_mixing_probability,
                                 (0, None, None, None, None, None, None))(
                                     traj_init[:, 0:2], gp.Z, gp.kernel,
                                     gp.mean_func, gp.q_mu, False, gp.q_sqrt)
    mixing_probs_opt = jax.vmap(single_mogpe_mixing_probability,
                                (0, None, None, None, None, None, None))(
                                    traj_opt[:, 0:2], gp.Z, gp.kernel,
                                    gp.mean_func, gp.q_mu, False, gp.q_sqrt)

    plot_3d_traj_col(fig,
                     ax,
                     traj_init,
                     zs=mixing_probs_init[:, 0:1],
                     color='c')
    plot_3d_traj_col(fig, ax, traj_opt, zs=mixing_probs_opt[:, 0:1], color='m')
    return fig, ax


def plot_3d_traj_metric_trace_init_and_opt(fig, ax, metric, traj_init,
                                           traj_opt):
    from ProbGeo.metric_tensor import gp_metric_tensor, metric_tensor_fn

    metric_tensor_init, _, _ = gp_metric_tensor(traj_init[:, 0:2],
                                                metric.gp.Z,
                                                metric.gp.kernel,
                                                mean_func=metric.gp.mean_func,
                                                f=metric.gp.q_mu,
                                                full_cov=True,
                                                q_sqrt=metric.gp.q_sqrt,
                                                cov_weight=metric.cov_weight)
    metric_tensor_opt, _, _ = gp_metric_tensor(traj_opt[:, 0:2],
                                               metric.gp.Z,
                                               metric.gp.kernel,
                                               mean_func=metric.gp.mean_func,
                                               f=metric.gp.q_mu,
                                               full_cov=True,
                                               q_sqrt=metric.gp.q_sqrt,
                                               cov_weight=metric.cov_weight)

    metric_trace_init = np.trace(metric_tensor_init, axis1=1, axis2=2)
    metric_trace_opt = np.trace(metric_tensor_opt, axis1=1, axis2=2)

    plot_3d_traj_col(fig, ax, traj_init, zs=metric_trace_init, color='c')
    plot_3d_traj_col(fig, ax, traj_opt, zs=metric_trace_opt, color='m')
    return fig, ax


def plot_3d_traj_col(fig, ax, traj, zs, color='k'):
    # ax.scatter(traj[:, 0], traj[:, 1], zs=zs, marker='x', color='k')
    # ax.plot3D(traj[:, 0], traj[:, 1], zs=zs.flatten(), color='k')
    ax.scatter(traj[:, 0], traj[:, 1], zs=0, zdir='z', color=color, marker='x')
    ax.scatter(traj[:, 0], 3, zs=zs, zdir='z', color=color, marker='x')
    ax.scatter(-2, traj[:, 1], zs=zs, zdir='z', color=color, marker='x')
    ax.plot3D(traj[:, 0],
              traj[:, 1],
              zs=0 * np.ones([*traj[:, 0].shape]),
              zdir='z',
              color=color)
    ax.plot3D(traj[:, 0],
              3 * np.ones([*traj[:, 0].shape]),
              zs=zs.flatten(),
              zdir='z',
              color=color)
    ax.plot3D(-2 * np.ones([*traj[:, 0].shape]),
              traj[:, 1],
              zs=zs.flatten(),
              zdir='z',
              color=color)
    return fig, ax


def plot_3d_traj(fig, ax, traj, zs):
    # ax.scatter(traj[:, 0], traj[:, 1], zs=zs, marker='x', color='k')
    # ax.plot3D(traj[:, 0], traj[:, 1], zs=zs.flatten(), color='k')
    ax.scatter(traj[:, 0], traj[:, 1], zs=0, zdir='z', color='k', marker='x')
    ax.scatter(traj[:, 0], 3, zs=zs, zdir='z', color='c', marker='x')
    ax.scatter(-2, traj[:, 1], zs=zs, zdir='z', color='m', marker='x')
    ax.plot3D(traj[:, 0],
              traj[:, 1],
              zs=0 * np.ones([*traj[:, 0].shape]),
              zdir='z',
              color='k')
    ax.plot3D(traj[:, 0],
              3 * np.ones([*traj[:, 0].shape]),
              zs=zs.flatten(),
              zdir='z',
              color='c')
    ax.plot3D(-2 * np.ones([*traj[:, 0].shape]),
              traj[:, 1],
              zs=zs.flatten(),
              zdir='z',
              color='m')
    return fig, ax


def plot_gp_and_start_end(gp, solver):
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(Xnew,
                         gp.X,
                         kernels=gp.kernel,
                         mean_funcs=gp.mean_func,
                         f=gp.Y,
                         q_sqrt=gp.q_sqrt,
                         full_cov=False)
    fig, axs = plot_mean_and_var(xx, yy, mu, var)

    for ax in axs:
        fig, ax = plot_start_and_end_pos(fig, ax, solver)
    return fig, axs


def plot_start_and_end_pos(fig, ax, solver):
    ax.scatter(solver.pos_init[0], solver.pos_init[1], marker='x', color='k')
    ax.scatter(solver.pos_end_targ[0],
               solver.pos_end_targ[1],
               color='k',
               marker='x')
    ax.annotate(
        "Start $\mathbf{x}_0$",
        (solver.pos_init[0], solver.pos_init[1]),
        horizontalalignment='left',
        verticalalignment='top',
    )
    ax.annotate(
        "End $\mathbf{x}_f$",
        (solver.pos_end_targ[0], solver.pos_end_targ[1]),
        horizontalalignment='left',
        verticalalignment='top',
    )
    return fig, ax


def plot_svgp_and_start_end(gp, solver, traj_opt=None):
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict
    params = {
        'text.usetex': True,
        'text.latex.preamble':
        ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
    }
    plt.rcParams.update(params)

    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(Xnew,
                         gp.Z,
                         kernels=gp.kernel,
                         mean_funcs=gp.mean_func,
                         f=gp.q_mu,
                         q_sqrt=gp.q_sqrt,
                         full_cov=False)
    print('mu var')
    # print(mu.shape)
    # print(var.shape)
    # mu = mu[0:1, :, :]
    # var = var[0:1, :]
    # mu = mu[1:2, :, :]
    # var = var[1:2, :]
    print(mu.shape)
    print(var.shape)
    fig, axs = plot_mean_and_var(xx,
                                 yy,
                                 mu,
                                 var,
                                 llabel='$\mathbb{E}[h^{(1)}]$',
                                 rlabel='$\mathbb{V}[h^{(1)}]$')

    for ax in axs:
        fig, ax = plot_start_and_end_pos(fig, ax, solver)
        plot_omitted_data(fig, ax, color='k')
        # ax.scatter(gp.X[:, 0], gp.X[:, 1])
        plot_traj(fig,
                  ax,
                  solver.state_guesses,
                  color=color_init,
                  label='Initial trajectory')
        if traj_opt is not None:
            plot_traj(fig,
                      ax,
                      traj_opt,
                      color=color_opt,
                      label='Optimised trajectory')
    axs[0].legend()
    return fig, axs


def plot_svgp_jacobian_mean(gp, solver, traj_opt=None):
    from ProbGeo.visualisation.gp import plot_mean_and_var
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_jacobian, gp_predict
    params = {
        'text.usetex': True,
        'text.latex.preamble':
        ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
    }
    plt.rcParams.update(params)

    Xnew, xx, yy = create_grid(gp.X, N=961)
    mu, var = gp_predict(Xnew,
                         gp.Z,
                         kernels=gp.kernel,
                         mean_funcs=gp.mean_func,
                         f=gp.q_mu,
                         q_sqrt=gp.q_sqrt,
                         full_cov=False)

    def gp_jacobian_all(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return gp_jacobian(x,
                           gp.Z,
                           gp.kernel,
                           gp.mean_func,
                           f=gp.q_mu,
                           q_sqrt=gp.q_sqrt,
                           full_cov=False)

    mu_j, var_j = jax.vmap(gp_jacobian_all, in_axes=(0))(Xnew)
    print('gp jacobain mu var')
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
        llabel='$\mathbb{E}[h^{(1)}]$',
        rlabel='$\mathbb{V}[h^{(1)}]$')

    for ax in axs:
        ax.quiver(Xnew[:, 0], Xnew[:, 1], mu_j[:, 0], mu_j[:, 1], color='k')

        fig, ax = plot_start_and_end_pos(fig, ax, solver)
        plot_omitted_data(fig, ax, color='k')
        # ax.scatter(gp.X[:, 0], gp.X[:, 1])
        plot_traj(fig,
                  ax,
                  solver.state_guesses,
                  color=color_init,
                  label='Initial trajectory')
        if traj_opt is not None:
            plot_traj(fig,
                      ax,
                      traj_opt,
                      color=color_opt,
                      label='Optimised trajectory')
    axs[0].legend()
    return fig, axs


def plot_svgp_jacobian_var(gp, solver, traj_opt=None):
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.visualisation.gp import plot_contourf
    from ProbGeo.gp import gp_jacobian, gp_predict
    params = {
        'text.usetex': True,
        'text.latex.preamble':
        ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
    }
    plt.rcParams.update(params)
    input_dim = gp.X.shape[1]

    Xnew, xx, yy = create_grid(gp.X, N=961)

    def gp_jacobian_all(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return gp_jacobian(x,
                           gp.Z,
                           gp.kernel,
                           gp.mean_func,
                           f=gp.q_mu,
                           q_sqrt=gp.q_sqrt,
                           full_cov=False)

    mu_j, var_j = jax.vmap(gp_jacobian_all, in_axes=(0))(Xnew)
    print('gp jacobain mu var')
    print(mu_j.shape)
    print(var_j.shape)
    # mu = np.prod(mu, 1)
    # var = np.diagonal(var, axis1=-2, axis2=-1)
    # var = np.prod(var, 1)

    fig, axs = plt.subplots(input_dim, input_dim)
    for i in range(input_dim):
        for j in range(input_dim):
            plot_contourf(fig,
                          axs[i, j],
                          xx,
                          yy,
                          var_j[:, i, j],
                          label='$\Sigma_J(\mathbf{x})$')
            axs[i, j].set_xlabel('$x$')
            axs[i, j].set_ylabel('$y$')

    return fig, axs


def plot_svgp_metric_and_start_end(metric, solver, traj_opt=None):
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.visualisation.gp import plot_contourf
    from ProbGeo.metric_tensor import gp_metric_tensor, metric_tensor_fn
    params = {
        'text.usetex': True,
        'text.latex.preamble':
        ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
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
        cov_weight=metric.cov_weight)
    print('metric yo yo')
    print(metric_tensor.shape)

    fig, axs = plt.subplots(input_dim, input_dim)
    for i in range(input_dim):
        for j in range(input_dim):
            plot_contourf(fig,
                          axs[i, j],
                          xx,
                          yy,
                          metric_tensor[:, i, j],
                          label='$G(\mathbf{x})$')
            axs[i, j].set_xlabel('$x$')
            axs[i, j].set_ylabel('$y$')

    return fig, axs


def plot_epistemic_var_vs_time(gp, solver, traj_init, traj_opt=None):
    from ProbGeo.gp import gp_predict
    params = {
        'text.usetex': True,
        'text.latex.preamble':
        ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
    }
    plt.rcParams.update(params)

    _, var_init = gp_predict(traj_init[:, 0:2],
                             gp.Z,
                             kernels=gp.kernel,
                             mean_funcs=gp.mean_func,
                             f=gp.q_mu,
                             q_sqrt=gp.q_sqrt,
                             full_cov=False)
    var_opt = 0.
    if traj_opt is not None:
        _, var_opt = gp_predict(traj_opt[:, 0:2],
                                gp.Z,
                                kernels=gp.kernel,
                                mean_funcs=gp.mean_func,
                                f=gp.q_mu,
                                q_sqrt=gp.q_sqrt,
                                full_cov=False)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$\mathbb{V}[h^{(1)}]$')

    ax.plot(solver.times,
            var_init[0, :],
            color=color_init,
            label='Initial trajectory')
    if traj_opt is not None:
        ax.plot(solver.times,
                var_opt[0, :],
                color=color_opt,
                label='Optimised trajectory')
    ax.legend()
    sum_var_init = np.sum(var_init)
    sum_var_opt = np.sum(var_opt)
    print('Sum epistemic var init = ', sum_var_init)
    print('Sum epistemic var opt = ', sum_var_opt)
    return fig, ax


def plot_aleatoric_var_vs_time(gp, solver, traj_init, traj_opt=None):
    # from ProbGeo.gp import gp_predict
    from ProbGeo.mogpe import single_mogpe_mixing_probability
    params = {
        'text.usetex': True,
        'text.latex.preamble':
        ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
    }
    plt.rcParams.update(params)

    mixing_probs_init = jax.vmap(single_mogpe_mixing_probability,
                                 (0, None, None, None, None, None, None))(
                                     traj_init[:, 0:2], gp.Z, gp.kernel,
                                     gp.mean_func, gp.q_mu, False, gp.q_sqrt)
    if mixing_probs_init.shape[-1] == 1:
        mixing_probs_init = np.concatenate(
            [mixing_probs_init, 1 - mixing_probs_init], -1)
    if traj_opt is not None:
        mixing_probs_opt = jax.vmap(
            single_mogpe_mixing_probability,
            (0, None, None, None, None, None, None))(traj_opt[:, 0:2], gp.Z,
                                                     gp.kernel, gp.mean_func,
                                                     gp.q_mu, False, gp.q_sqrt)
        if mixing_probs_opt.shape[-1] == 1:
            mixing_probs_opt = np.concatenate(
                [mixing_probs_opt, 1 - mixing_probs_opt], -1)

    noise_vars = np.array(gp.noise_vars).reshape(-1, 1)
    var_init = mixing_probs_init @ noise_vars

    var_opt = 0
    if traj_opt is not None:
        var_opt = mixing_probs_opt @ noise_vars
        print('var opt')
        print(var_opt.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.8))

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$\sum_{k=1}^K\Pr(\\alpha=k|\mathbf{x}) (\sigma^{(k)})^2$')

    ax.plot(solver.times,
            var_init,
            color=color_init,
            label='Initial trajectory')
    if traj_opt is not None:
        ax.plot(solver.times,
                var_opt,
                color=color_opt,
                label='Optimised trajectory')
    ax.legend()
    sum_var_init = np.sum(var_init)
    sum_var_opt = np.sum(var_opt)
    print('Sum aleatoric var init = ', sum_var_init)
    print('Sum aleatoric var opt = ', sum_var_opt)
    return fig, ax


def plot_traj(fig, axs, traj, color='k', label=''):
    try:
        for ax in axs:
            ax.scatter(traj[:, 0], traj[:, 1], marker='x', color=color)
            ax.plot(traj[:, 0],
                    traj[:, 1],
                    marker='x',
                    color=color,
                    label=label)
    except:
        axs.scatter(traj[:, 0], traj[:, 1], marker='x', color=color)
        axs.plot(traj[:, 0], traj[:, 1], marker='x', color=color, label=label)
    return fig, axs


def plot_mixing_prob_and_start_end(gp, solver, traj_opt=None):
    from ProbGeo.visualisation.gp import plot_contourf
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.mogpe import mogpe_mixing_probability, single_mogpe_mixing_probability

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)

    mixing_probs = jax.vmap(single_mogpe_mixing_probability,
                            (0, None, None, None, None, None, None))(
                                Xnew, gp.Z, gp.kernel, gp.mean_func, gp.q_mu,
                                False, gp.q_sqrt)
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
        label='$\Pr(\\alpha=1 | \mathbf{x})$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    plot_omitted_data(fig, ax, color='k')

    plot_start_and_end_pos(fig, ax, solver)

    plot_traj(fig,
              ax,
              solver.state_guesses,
              color=color_init,
              label='Initial trajectory')
    if traj_opt is not None:
        plot_traj(fig,
                  ax,
                  traj_opt,
                  color=color_opt,
                  label='Optimised trajectory')
    ax.legend()

    return fig, ax


def plot_omitted_data(fig,
                      ax,
                      x1_low=-0.5,
                      x2_low=-3.,
                      x1_high=1.,
                      x2_high=1.,
                      color='r'):
    x1_low = -1.
    x2_low = -1.
    x1_high = 1.
    x2_high = 2.5
    ax.add_patch(
        patches.Rectangle((x1_low, x2_low),
                          x1_high - x1_low,
                          x2_high - x2_low,
                          edgecolor=color,
                          facecolor='red',
                          hatch='/',
                          label='No observations',
                          fill=False))

    return fig, ax


def plot_svgp_mixing_prob_and_start_end(gp, solver, traj_opt=None):
    from ProbGeo.visualisation.gp import plot_contourf
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.mogpe import mogpe_mixing_probability

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)
    mixing_probs = mogpe_mixing_probability(Xnew,
                                            gp.Z,
                                            gp.kernel,
                                            mean_func=gp.mean_func,
                                            f=gp.q_mu,
                                            q_sqrt=gp.q_sqrt,
                                            full_cov=False)
    fig, ax = plt.subplots(1, 1)
    plot_contourf(fig, ax, xx, yy, mixing_probs[:, 0:1])

    plot_start_and_end_pos(fig, ax, solver)
    plot_traj(fig,
              ax,
              solver.state_guesses,
              color=color_init,
              label='Initial trajectory')
    if traj_opt is not None:
        plot_traj(fig,
                  ax,
                  traj_opt,
                  color=color_opt,
                  label='Optimised trajectory')
    ax.legend()
    return fig, ax


def plot_svgp_metric_trace_and_start_end(metric, solver, traj_opt=None):
    from ProbGeo.visualisation.gp import plot_contourf
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.metric_tensor import gp_metric_tensor, metric_tensor_fn

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
        cov_weight=metric.cov_weight)

    metric_trace = np.trace(metric_tensor, axis1=1, axis2=2)
    fig, ax = plt.subplots(1, 1)
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    surf_traceG = plot_contourf(fig,
                                ax,
                                xx,
                                yy,
                                metric_trace,
                                label='Tr$(G(\mathbf{x}))$')
    plot_start_and_end_pos(fig, ax, solver)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plot_traj(fig,
              ax,
              solver.state_guesses,
              color=color_init,
              label='Initial trajectory')
    if traj_opt is not None:
        plot_traj(fig,
                  ax,
                  traj_opt,
                  color=color_opt,
                  label='Optimised trajectory')
    ax.legend()
    return fig, ax


class FakeGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    X, Y, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    mean_func = 0.
    q_sqrt = None


class FakeSVGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse, load_data_and_init_kernel_mogpe
    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../models/saved_models/params_from_model.npz')
    # filename='../models/saved_models/params_fake_sparse_26-08.npz')
    # filename='../models/saved_models/params_from_model.npz')
    # filename='../models/saved_models/params_fake_sparse_20-08.npz')
    # filename='../models/saved_models/params_fake_sparse_20-08_2.npz')
    # Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_mogpe(
    #     save_model_dir=
    #     "../models/saved_models/quadcopter/09-10-115323-param_dict.pickle")
    # "../models/saved_models/quadcopter/09-08-153416-param_dict.pickle")
    # "../models/saved_models/quadcopter/09-08-144846-param_dict.pickle")


class FakeSVGPQuad:
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse, load_data_and_init_kernel_mogpe
    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../models/saved_models/params_from_model.npz')
    Z, q_mu, q_sqrt, kernel, mean_func, noise_vars = load_data_and_init_kernel_mogpe(
        '0',
        # save_model_dir=
        # "../models/saved_models/quadcopter/10-16-101754-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-02-104545-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-21-162809-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-21-180619-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-22-101801-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-22-113421-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-22-163319-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-23-115309-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-23-123636-param_dict.pickle")
        # "../models/saved_models/quadcopter/10-23-140423-param_dict.pickle")
        "../models/saved_models/quadcopter/10-23-160809-param_dict.pickle")
    # "../models/saved_models/quadcopter/10-23-092109-param_dict.pickle")
    # "../models/saved_models/quadcopter/10-23-094709-param_dict.pickle")
    # "../models/saved_models/quadcopter/10-23-103501-param_dict.pickle")
    # "../models/saved_models/quadcopter/10-23-112338-param_dict.pickle")
    # "../models/saved_models/quadcopter/09-10-115323-param_dict.pickle")
    # "../models/saved_models/quadcopter/09-08-153416-param_dict.pickle")
    # "../models/saved_models/quadcopter/09-08-144846-param_dict.pickle")


class FakeGPMetric:
    gp = FakeGP()
    # cov_weight = 38.
    # cov_weight = 1.
    # cov_weight = 0.35
    cov_weight = 2.
    # cov_weight = 0.01
    # cov_weight = 0.3
    full_cov = True
    q_sqrt = None
    jitter = 1e-4
    white = True
    metric_fn_kwargs = {
        "X": gp.X,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.Y,
        "full_cov": full_cov,
        "q_sqrt": q_sqrt,
        "cov_weight": cov_weight,
        "jitter": jitter,
        "white": white
    }


class FakeSVGPMetric:
    gp = FakeSVGP()
    # cov_weight = 38.
    cov_weight = 50.
    # cov_weight = 500.
    # cov_weight = 0.
    # cov_weight = 0.3
    full_cov = True
    jitter = 1e-4
    white = True
    metric_fn_kwargs = {
        "X": gp.Z,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.q_mu,
        "full_cov": full_cov,
        "q_sqrt": gp.q_sqrt,
        "cov_weight": cov_weight,
        "jitter": jitter,
        "white": white
    }


class FakeSVGPMetricQuad:
    gp = FakeSVGPQuad()
    # cov_weight = 38.
    cov_weight = 1.
    cov_weight = 5.8
    cov_weight = 20.
    # cov_weight = 50.
    # cov_weight = 70.
    # cov_weight = 0.3
    cov_weight = 0.5
    # cov_weight = 0.01
    full_cov = True
    jitter = 1e-4
    white = True
    metric_fn_kwargs = {
        "X": gp.Z,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.q_mu,
        "full_cov": full_cov,
        "q_sqrt": gp.q_sqrt,
        "cov_weight": cov_weight,
        "jitter": jitter,
        "white": white
    }
    # from ProbGeo.metric_tensor import gp_metric_tensor
    # fun = gp_metric_tensor


class FakeGPMetricProb:
    # from ProbGeo.mogpe import single_mogpe_mixing_probability
    from ProbGeo.mogpe import single_mogpe_mixing_probability
    gp = FakeGP()
    # cov_weight = 38.
    # cov_weight = 1.
    # cov_weight = 0.35
    full_cov = True
    q_sqrt = None
    jitter = 1e-4
    white = True
    fun_kwargs = {
        "X": gp.X,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.Y,
        "full_cov": full_cov,
        "q_sqrt": q_sqrt,
        "jitter": jitter,
        "white": white
    }
    fun = single_mogpe_mixing_probability
    metric_tensor_fn_kwargs = {'fun': fun, 'fun_kwargs': fun_kwargs}


class FakeSVGPMetricProb:
    from ProbGeo.mogpe import single_mogpe_mixing_probability
    gp = FakeSVGP()
    full_cov = True
    jitter = 1e-4
    white = True
    fun_kwargs = {
        "X": gp.Z,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.q_mu,
        "full_cov": full_cov,
        "q_sqrt": gp.q_sqrt,
        "jitter": jitter,
        "white": white
    }
    fun = single_mogpe_mixing_probability
    metric_tensor_fn_kwargs = {'fun': fun, 'fun_kwargs': fun_kwargs}


class FakeSVGPMetricProbQuad:
    from ProbGeo.mogpe import single_mogpe_mixing_probability
    gp = FakeSVGPQuad()
    full_cov = True
    jitter = 1e-4
    white = True
    fun_kwargs = {
        "X": gp.Z,
        "kernel": gp.kernel,
        "mean_func": gp.mean_func,
        "f": gp.q_mu,
        "full_cov": full_cov,
        "q_sqrt": gp.q_sqrt,
        "jitter": jitter,
        "white": white
    }
    fun = single_mogpe_mixing_probability
    metric_tensor_fn_kwargs = {'fun': fun, 'fun_kwargs': fun_kwargs}


class FakeCollocation:
    num_col_points = 20
    # num_col_points = 5
    input_dim = 2
    t_init = 0.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
    print(times.shape)
    pos_init = np.array([2.8, -2.5])
    pos_end_targ = np.array([-1.5, 2.8])
    vel_init_guess = np.array([-0.5641, 0.395])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


class FakeCollocationSVGP:
    num_col_points = 20
    input_dim = 2
    t_init = 0.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
    pos_init = np.array([1.7, -0.7])
    pos_end_targ = np.array([0.1, 1.8])
    vel_init_guess = np.array([-0.05, -0.03])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


class FakeCollocationSVGPQuad:
    # num_col_points = 20
    # num_col_points = 100
    num_col_points = 10
    input_dim = 2
    # t_init = 0.
    t_init = -1.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
    # pos_init = np.array([-1.7, -0.7])
    # pos_end_targ = np.array([0.1, 1.8])
    # pos_init = np.array([-1.4, -1.5])
    # pos_init = np.array([-1.5, -1.5])
    pos_init = np.array([-0.5, -1.5])
    pos_init = np.array([1.5, -2.])
    pos_init = np.array([-0.5, -2.5])
    pos_init = np.array([1., -2.5])
    pos_init = np.array([-1., -2.])
    pos_init = np.array([0.5, -2.])
    pos_init = np.array([0.7, -2.3])
    # pos_init = np.array([-2., -0.5])
    # pos_init = np.array([-1.4, -1.])
    # pos_mid = np.array([-1., 1.5])
    # pos_mid = np.array([-.5, 1.])
    pos_mid = np.array([-.5, 0.5])
    pos_mid = np.array([1.5, 1.5])
    # pos_mid = np.array([-1.5, 1.5])
    # pos_mid = np.array([-1., 1.])
    # pos_end_targ = np.array([1.5, 1.9])
    # pos_end_targ = np.array([2., 2.5])
    pos_end_targ = np.array([2., 1.5])
    pos_end_targ = np.array([1., 2.2])
    # pos_end_targ = np.array([1., 1.8])
    # pos_end_targ = np.array([1.5, 2.])
    # pos_end_targ = np.array([1.5, 2.5])
    pos11_guesses = np.linspace(pos_init[0], pos_mid[0],
                                int(num_col_points / 2))
    pos21_guesses = np.linspace(pos_init[1], pos_mid[1],
                                int(num_col_points / 2))
    pos12_guesses = np.linspace(pos_mid[0], pos_end_targ[0],
                                int(num_col_points / 2))
    pos22_guesses = np.linspace(pos_mid[1], pos_end_targ[1],
                                int(num_col_points / 2))
    pos1_guesses = np.concatenate([pos11_guesses, pos12_guesses])
    pos2_guesses = np.concatenate([pos21_guesses, pos22_guesses])
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)

    # vel_init_guess = np.array([0.05, 0.03])
    # vel_init_guess = np.array([0.15, 0.13])
    # vel_init_guess = np.array([0.005, 0.003])
    vel_init_guess = np.array([0.0000005, 0.0000003])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


class FakeCollocationProb:
    num_col_points = 50
    # num_col_points = 10
    input_dim = 2
    t_init = 0.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
    pos_init = np.array([2.8, -2.5])
    pos_end_targ = np.array([-1.5, 2.8])
    vel_init_guess = np.array([-0.5641, 0.395])
    # vel_init_guess = np.array([-0.0005641, 0.000395])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


class FakeCollocationProbSVGP:
    # num_col_points = 50
    num_col_points = 5
    input_dim = 2
    t_init = 0.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
    # pos_init = np.array([2.8, -2.5])
    # pos_end_targ = np.array([-1.5, 2.8])
    pos_init = np.array([-1.4, -1.])
    pos_end_targ = np.array([1.5, 1.9])
    # vel_init_guess = np.array([-0.5641, 0.395])
    vel_init_guess = np.array([-1.5641, 1.395])
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


class FakeCollocationProbSVGPQuad:
    # num_col_points = 50
    # num_col_points = 80
    num_col_points = 5
    input_dim = 2
    t_init = 0.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
    # pos_init = np.array([-1.4, 1.])
    pos_init = np.array([-1.4, -1.])
    pos_end_targ = np.array([1.5, 1.9])
    # vel_init_guess = np.array([-0.5641, 0.395])
    # vel_init_guess = np.array([0.5641, 0.395])
    vel_init_guess = np.array([0.005641, 0.000395])
    # vel_init_guess = np.array([1.5641, 1.395])
    vel_end_guess = np.array([0.05641, 0.0395])
    vel_init_guess = np.array([0.00, 0.00])
    vel_end_guess = np.array([0.0, 0.0])
    vel1_guesses = np.linspace(vel_init_guess[0], vel_end_guess[0],
                               num_col_points)
    vel2_guesses = np.linspace(vel_init_guess[1], vel_end_guess[1],
                               num_col_points)
    pos1_guesses = np.linspace(pos_init[0], pos_end_targ[0], num_col_points)
    pos2_guesses = np.linspace(pos_init[1], pos_end_targ[1], num_col_points)
    pos_guesses = np.stack([pos1_guesses, pos2_guesses], -1)
    vel_guesses = np.stack([vel1_guesses, vel2_guesses], -1)
    vel_init_guess = np.array([0.01, 0.01])
    vel_guesses = np.broadcast_to(vel_init_guess, (num_col_points, input_dim))
    state_guesses = np.concatenate([pos_guesses, vel_guesses], -1)


def test_collocation():
    from ProbGeo.collocation import collocation, collocation_no_constraint, collocation_root
    from ProbGeo.metric_tensor import gp_metric_tensor
    solver = FakeCollocation()
    metric = FakeGPMetric()
    metric_fn = gp_metric_tensor
    geodesic_traj = collocation(solver.state_guesses, solver.pos_init,
                                solver.pos_end_targ, metric_fn,
                                metric.metric_fn_kwargs, solver.times)
    # geodesic_traj = collocation_no_constraint(solver.state_guesses,
    #                                           solver.pos_init,
    #                                           solver.pos_end_targ, metric_fn,
    #                                           metric.metric_fn_kwargs,
    #                                           solver.times)
    # geodesic_traj = collocation_root(solver.state_guesses, solver.pos_init,
    #                                  solver.pos_end_targ, metric_fn,
    #                                  metric.metric_fn_kwargs, solver.times)
    return geodesic_traj


def test_collocation_svgp():
    from ProbGeo.collocation import collocation
    from ProbGeo.metric_tensor import gp_metric_tensor
    solver = FakeCollocationSVGP()
    metric = FakeSVGPMetric()
    metric_fn = gp_metric_tensor
    geodesic_traj = collocation(solver.state_guesses, solver.pos_init,
                                solver.pos_end_targ, metric_fn,
                                metric.metric_fn_kwargs, solver.times)
    return geodesic_traj


def test_collocation_svgp_quad():
    from ProbGeo.collocation import collocation, collocation_root, collocation_
    from ProbGeo.metric_tensor import gp_metric_tensor
    solver = FakeCollocationSVGPQuad()
    metric = FakeSVGPMetricQuad()
    metric_fn = gp_metric_tensor
    geodesic_traj = collocation(solver.state_guesses, solver.pos_init,
                                solver.pos_end_targ, metric_fn,
                                metric.metric_fn_kwargs, solver.times)
    # geodesic_traj = collocation_root(solver.state_guesses, solver.pos_init,
    #                                  solver.pos_end_targ, metric_fn,
    #                                  metric.metric_fn_kwargs, solver.times)
    # geodesic_traj = collocation_(solver.pos_init, solver.pos_end_targ,
    #                              metric_fn, metric.metric_fn_kwargs,
    #                              solver.times)
    return geodesic_traj


def test_collocation_prob():
    from ProbGeo.metric_tensor import metric_tensor_fn
    from ProbGeo.collocation import collocation, collocation_root
    # solver = FakeODESolverProb()
    metric = FakeGPMetricProb()
    metric_fn = metric_tensor_fn
    solver = FakeCollocationProb()
    geodesic_traj = collocation(solver.state_guesses, solver.pos_init,
                                solver.pos_end_targ, metric_fn,
                                metric.metric_tensor_fn_kwargs, solver.times)
    # geodesic_traj = collocation_root(solver.state_guesses, solver.pos_init,
    #                                  solver.pos_end_targ, metric_fn,
    #                                  metric.metric_tensor_fn_kwargs,
    #                                  solver.times)
    return geodesic_traj


def test_collocation_prob_svgp():
    from ProbGeo.metric_tensor import metric_tensor_fn
    from ProbGeo.collocation import collocation
    metric = FakeSVGPMetricProb()
    metric_fn = metric_tensor_fn
    solver = FakeCollocationProbSVGP()
    geodesic_traj = collocation(solver.state_guesses, solver.pos_init,
                                solver.pos_end_targ, metric_fn,
                                metric.metric_tensor_fn_kwargs, solver.times)
    return geodesic_traj


def test_collocation_prob_svgp_quad():
    from ProbGeo.metric_tensor import metric_tensor_fn
    from ProbGeo.collocation import collocation, collocation_no_constraint, collocation_root
    metric = FakeSVGPMetricProbQuad()
    metric_fn = metric_tensor_fn
    solver = FakeCollocationProbSVGPQuad()
    # geodesic_traj = collocation(solver.state_guesses, solver.pos_init,
    #                             solver.pos_end_targ, metric_fn,
    #                             metric.metric_tensor_fn_kwargs, solver.times)
    # geodesic_traj = collocation_no_constraint(solver.state_guesses,
    #                                           solver.pos_init,
    #                                           solver.pos_end_targ, metric_fn,
    #                                           metric.metric_tensor_fn_kwargs,
    #                                           solver.times)
    geodesic_traj = collocation_root(solver.state_guesses, solver.pos_init,
                                     solver.pos_end_targ, metric_fn,
                                     metric.metric_tensor_fn_kwargs,
                                     solver.times)
    return geodesic_traj


def test_and_plot_collocation():
    gp = FakeGP()
    solver = FakeCollocation()
    fig, axs = plot_gp_and_start_end(gp, solver)
    # plt.show()
    geodesic_traj = test_collocation()
    fig, axs = plot_traj(fig, axs, geodesic_traj)
    plt.show()


def test_and_plot_collocation_svgp():
    gp = FakeSVGP()
    solver = FakeCollocationSVGP()
    fig, axs = plot_svgp_and_start_end(gp, solver)
    plt.show()
    fig, axs = plot_svgp_and_start_end(gp, solver)
    geodesic_traj = test_collocation_svgp()
    fig, axs = plot_traj(fig, axs, geodesic_traj)
    plt.show()


def create_save_dir():
    import pathlib
    import time
    daystr = time.strftime("%d-%m-%Y")
    timestr = time.strftime("%H%M%S")
    dir_name = "../reports/figures/" + daystr + "/" + timestr + "/"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    return dir_name


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
        'text.usetex': True,
    }
    plt.rcParams.update(params)

    dir_name = create_save_dir()
    fig, ax = plot_mixing_prob_and_start_end(gp, solver, solver.state_guesses)
    # plot_traj(fig, ax, solver.state_guesses)
    save_name = dir_name + "mixing_prob_2d_init_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    fig, axs = plot_svgp_and_start_end(gp, solver)
    # plot_traj(fig, axs, solver.state_guesses)
    save_name = dir_name + "svgp_2d_init_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    fig, axs = plot_epistemic_var_vs_time(gp,
                                          solver,
                                          traj_init=solver.state_guesses,
                                          traj_opt=None)
    save_name = dir_name + "epistemic_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    plot_aleatoric_var_vs_time(gp,
                               solver,
                               traj_init=solver.state_guesses,
                               traj_opt=None)
    save_name = dir_name + "aleatoric_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    fig, ax = plot_svgp_metric_trace_and_start_end(metric, solver)
    plot_traj(fig, ax, solver.state_guesses)
    save_name = dir_name + "metric_trace_2d_init_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    fig, axs = plot_svgp_jacobian_mean(gp,
                                       solver,
                                       traj_opt=solver.state_guesses)
    fig, axs = plot_svgp_jacobian_var(gp,
                                      solver,
                                      traj_opt=solver.state_guesses)

    fig, axs = plot_svgp_metric_and_start_end(metric,
                                              solver,
                                              traj_opt=solver.state_guesses)

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
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # init and opt trajectories over mixing probability
    fig, ax = plot_mixing_prob_and_start_end(gp,
                                             solver,
                                             traj_opt=geodesic_traj)
    save_name = dir_name + "mixing_prob_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # init and opt trajectories over metric trace
    fig, ax = plot_svgp_metric_trace_and_start_end(metric,
                                                   solver,
                                                   traj_opt=geodesic_traj)
    save_name = dir_name + "metric_trace_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # prob vs time
    plot_mixing_prob_over_time(gp,
                               solver,
                               traj_init=solver.state_guesses,
                               traj_opt=geodesic_traj)
    save_name = dir_name + "mixing_prob_vs_time.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    # plot metric trace vs time
    plot_metirc_trace_over_time(metric,
                                solver,
                                traj_init=solver.state_guesses,
                                traj_opt=geodesic_traj)
    save_name = dir_name + "metric_trace_vs_time.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # plot epistemic uncertainty vs time
    fig, axs = plot_epistemic_var_vs_time(gp,
                                          solver,
                                          traj_init=solver.state_guesses,
                                          traj_opt=geodesic_traj)
    save_name = dir_name + "epistemic_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    plot_aleatoric_var_vs_time(gp,
                               solver,
                               traj_init=solver.state_guesses,
                               traj_opt=geodesic_traj)
    save_name = dir_name + "aleatoric_var_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)

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


def test_and_plot_collocation_prob():
    gp = FakeGP()
    solver = FakeCollocationProb()
    fig, ax = plot_mixing_prob_and_start_end(gp, solver)
    # plt.show()
    geodesic_traj = test_collocation_prob()
    fig, axs = plot_traj(fig, ax, geodesic_traj)
    plt.show()


def test_and_plot_collocation_prob_svgp():
    gp = FakeSVGP()
    solver = FakeCollocationProbSVGP()
    # fig, ax = plot_svgp_mixing_prob_and_start_end(gp, solver)
    # plt.show()
    fig, ax = plot_svgp_mixing_prob_and_start_end(gp, solver)
    geodesic_traj = test_collocation_prob_svgp()
    fig, axs = plot_traj(fig, ax, geodesic_traj)
    plt.show()


def test_and_plot_collocation_prob_svgp_quad():
    gp = FakeSVGPQuad()
    solver = FakeCollocationProbSVGPQuad()
    # fig, ax = plot_svgp_mixing_prob_and_start_end(gp, solver)
    # plt.show()
    fig, ax = plot_svgp_mixing_prob_and_start_end(gp, solver)
    geodesic_traj = test_collocation_prob_svgp_quad()
    fig, axs = plot_traj(fig, ax, geodesic_traj)
    plt.show()


if __name__ == "__main__":

    # test_and_plot_collocation()
    # test_and_plot_collocation_svgp()
    # test_and_plot_collocation_prob()
    # test_and_plot_collocation_prob_svgp()

    test_and_plot_collocation_svgp_quad()

    # test_and_plot_collocation_prob_svgp_quad()
