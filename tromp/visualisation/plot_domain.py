import jax
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import cm
from jax.config import config
config.update("jax_enable_x64", True)

from jax import numpy as np

params = {
    'axes.labelsize': 15,
    'font.size': 15,
    # 'legend.fontsize': 15, # used for mixing prob
    'legend.fontsize': 10,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'text.usetex': True,
    # 'text.latex.preamble': ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
}
params = {
    'axes.labelsize': 30,
    'font.size': 25,
    # 'legend.fontsize': 15, # used for mixing prob
    'legend.fontsize': 19,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'text.usetex': True,
    # 'text.latex.preamble': ["\\usepackage{amssymb}", "\\usepackage{amsmath}"],
}
plt.rcParams.update(params)
color_init = 'c'
color_opt = 'm'
color_opt_1 = 'm'
color_opt_2 = 'olive'
color_opts = [color_opt_1, color_opt_2]


def plot_start_and_end_pos(fig, ax, solver):
    fan_x = 2.5
    fan_y = -0.1
    fan_delta_x = 0.5
    fan_delta_y = 0.5
    ax.annotate(
        "Fan",
        (fan_x - 0.2, fan_y + 0.1),
        horizontalalignment='right',
        verticalalignment='top',
    )
    ax.plot([fan_x, fan_x - fan_delta_x], [fan_y, fan_y + fan_delta_y],
            color='k',
            linewidth=4)
    ax.plot([fan_x, fan_x - fan_delta_x], [fan_y, fan_y - fan_delta_y],
            color='k',
            linewidth=4)
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


def plot_domain_and_start_end(gp, solver, traj_opts=None, labels=None):
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.mogpe import mogpe_mixing_probability, single_mogpe_mixing_probability

    # plot original GP
    Xnew, xx, yy = create_grid(gp.X, N=961)

    mixing_probs = jax.vmap(single_mogpe_mixing_probability,
                            (0, None, None, None, None, None, None))(
                                Xnew, gp.Z, gp.kernel, gp.mean_func, gp.q_mu,
                                False, gp.q_sqrt)
    fig, ax = plt.subplots(1, 1)

    contf = ax.contourf(xx,
                        yy,
                        mixing_probs[:, 0:1].reshape(xx.shape),
                        cmap=cm.coolwarm,
                        levels=[0., 0.5, 1.0],
                        linewidth=0,
                        antialiased=False)
    # cbar = fig.colorbar(contf, shrink=0.5, aspect=5, ax=ax)
    # cbar.set_label('$Pr(\\alpha=1 | \mathbf{x})$')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    plot_omitted_data(fig, ax, color='k')
    plot_start_and_end_pos(fig, ax, solver)

    ax.annotate(
        "Mode 1",
        (-2.2, -1.5),
        horizontalalignment='left',
        verticalalignment='top',
    )
    ax.annotate(
        "Mode 2",
        (0.1, -0.2),
        horizontalalignment='left',
        verticalalignment='top',
    )

    # plot_traj(fig,
    #           ax,
    #           solver.state_guesses,
    #           color=color_init,
    #           label='Initial trajectory')
    # if traj_opts is not None:
    #     for traj, label, color in zip(traj_opts, labels, color_opts):
    #         plot_traj(fig, ax, traj, color=color, label=label)
    ax.legend(loc=3)

    return fig, ax


def plot_omitted_data(fig,
                      ax,
                      x1_low=-0.5,
                      x2_low=-3.,
                      x1_high=1.,
                      x2_high=1.,
                      color='r'):
    x1_low = -1.
    # x2_low = -1.
    x2_low = 0.
    x1_high = 0.9
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


class FakeSVGPQuad:
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse, load_data_and_init_kernel_mogpe
    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../../models/saved_models/params_from_model.npz')
    Z, q_mu, q_sqrt, kernel, mean_func, noise_vars = load_data_and_init_kernel_mogpe(
        '0',
        "../../models/saved_models/quadcopter/10-23-160809-param_dict.pickle")


class FakeSVGPMetricQuadLow:
    gp = FakeSVGPQuad()
    cov_weight = 0.5
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


class FakeSVGPMetricQuadHigh:
    gp = FakeSVGPQuad()
    cov_weight = 20.
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


class FakeCollocationSVGPQuad:
    # num_col_points = 20
    # num_col_points = 100
    num_col_points = 10
    input_dim = 2
    # t_init = 0.
    t_init = -1.
    t_end = 1.
    times = np.linspace(t_init, t_end, num_col_points)
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


def create_save_dir():
    import pathlib
    dir_name = "../../reports/figures/"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    return dir_name


if __name__ == "__main__":
    traj_1_filename = "../../reports/figures/26-10-2020/144218/geodesic_traj.npz"
    traj_2_filename = "../../reports/figures/26-10-2020/143053/geodesic_traj.npz"

    gp = FakeSVGPQuad()
    metric_low = FakeSVGPMetricQuadLow()
    metric_high = FakeSVGPMetricQuadHigh()
    metrics = [metric_low, metric_high]
    solver = FakeCollocationSVGPQuad()
    params = {
        'text.usetex': True,
    }
    plt.rcParams.update(params)

    dir_name = create_save_dir()
    geodesic_traj_1 = np.load(traj_1_filename)['arr_0']
    geodesic_traj_2 = np.load(traj_2_filename)['arr_0']
    print('geodesic')
    print(geodesic_traj_2)

    traj_opts = [geodesic_traj_1, geodesic_traj_2]
    labels = ['Optimised traj $\lambda=20$', 'Optimised traj $\lambda=0.5$']
    labels_metric = ['20', '0.5']
    linestyles = ['-', '--']
    # plot metric traces over time
    fig, axs = plot_domain_and_start_end(gp,
                                         solver,
                                         traj_opts=traj_opts,
                                         labels=labels_metric)
    plot_traj(fig, axs, solver.state_guesses, color='cyan', label='Initial traj')
    plot_traj(fig, axs, geodesic_traj_1, color='magenta', label=labels[0])
    plot_traj(fig, axs, geodesic_traj_2, color='olive', label=labels[1])
    save_name = dir_name + "domain_2d_traj_with_trajs.png"
    # save_name = dir_name + "domain_2d_traj.pdf"
    plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
