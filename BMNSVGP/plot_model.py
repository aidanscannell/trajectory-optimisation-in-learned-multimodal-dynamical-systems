import gpflow
import matplotlib.pyplot as plt
import numpy as np
import palettable
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_arrow(line,
              position=None,
              direction='right',
              size=15,
              color=None,
              alpha=1):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
        position = ydata.mean()
    # find closest index
    epsilon = 0.09
    cond1 = (ydata > position - epsilon)
    cond2 = (ydata < position + epsilon)
    cond = cond1 & cond2
    # start_ind = np.argmin(np.absolute(xdata - position))
    # start_ind = np.argmin(np.absolute(ydata - position))
    start_inds = np.where(cond)[0]
    print(start_inds)
    if direction == 'right':
        # end_ind = start_ind + 1
        end_inds = start_inds + 1
    else:
        # end_ind = start_ind - 1
        end_inds = start_inds - 1

    for start_ind, end_ind in zip(start_inds, end_inds):
        line.axes.annotate('',
                           xytext=(xdata[start_ind], ydata[start_ind]),
                           xy=(xdata[end_ind], ydata[end_ind]),
                           arrowprops=dict(arrowstyle="->", color=color),
                           size=size,
                           alpha=alpha)


def create_grid(X, N):
    x1_low, x1_high, x2_low, x2_high = X[:, 0].min(), X[:,
                                                        0].max(), X[:, 1].min(
                                                        ), X[:, 1].max()
    # x1_low, x1_high, x2_low, x2_high = -2., 3., -3., 3.
    sqrtN = int(np.sqrt(N))
    xx = np.linspace(x1_low, x1_high, sqrtN)
    yy = np.linspace(x2_low, x2_high, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    return xy, xx, yy


def plot_contourf(x,
                  y,
                  z,
                  fig=None,
                  ax=None,
                  tri=False,
                  clabel=False,
                  cbar_label='',
                  levels=None):
    # fig = plt.figure(figsize=(12, 4))  # no frame
    flag = False
    if fig is None:
        flag = True
        fig = plt.figure()  # no frame
        ax = fig.add_subplot(111)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22, left=0.09)

    if tri:
        contf = ax.tricontourf(x, y, z, 100, cmap=cmap)
        # antialiased=False,
        # levels=levels)
    else:
        contf = ax.contourf(x, y, z, cmap=cmap, levels=levels)

    if flag:
        cbar = fig.colorbar(
            contf,
            # shrink=0.5,
            # aspect=5,
            ax=ax)
        cbar.set_label(cbar_label)

    if clabel:
        ax.clabel(contf, fmt='%2.1f', colors='k')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)
    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(('outward', 5))
    # put the grid behind
    ax.set_axisbelow(True)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    # change xlim to set_xlim
    ax.set_xlim(-1.9, 2)
    ax.set_ylim(-2, 1.5)
    return contf, ax

    # contf = plot_contourf_ax(fig, ax, x, y, z, clabel, cbar_label, levels)
    # return plot_contourf_ax(fig, ax, x, y, z, clabel, cbar_label, levels)


def plot_mean_and_var(x,
                      y,
                      z_mu,
                      z_var,
                      title='',
                      clabel=False,
                      left_label='Mean',
                      right_label='Variance'):

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    contf_mu, ax_mu = plot_contourf(x,
                                    y,
                                    z_mu,
                                    fig,
                                    axs[0],
                                    clabel=clabel,
                                    cbar_label=left_label)
    divider = make_axes_locatable(axs[0])
    cax1 = divider.append_axes("top", size="5%", pad=0.05)
    cbar = fig.colorbar(contf_mu,
                        cax=cax1,
                        ax=axs[0],
                        orientation="horizontal")
    cbar.set_label(left_label)
    cax1.xaxis.set_ticks_position('top')
    cax1.xaxis.set_label_position('top')
    contf_var, ax_var = plot_contourf(x,
                                      y,
                                      z_var,
                                      fig,
                                      axs[1],
                                      clabel=clabel,
                                      cbar_label=right_label)
    divider = make_axes_locatable(axs[1])
    cax2 = divider.append_axes("top", size="5%", pad=0.05)
    cbar = fig.colorbar(contf_var,
                        cax=cax2,
                        ax=axs[1],
                        orientation="horizontal")
    cbar.set_label(right_label)
    cax2.xaxis.set_ticks_position('top')
    cax2.xaxis.set_label_position('top')
    # cbar = fig.colorbar(contf_var, ax=axs[1], orientation="horizontal")
    # cbar.set_label(right_label)
    plt.suptitle(title)
    return fig, (ax_mu, ax_var)


# bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
cmap = palettable.colorbrewer.sequential.BuPu_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.YlOrBr_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.YlGnBu_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.YlGn_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.Reds_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.RdPu_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.Purples_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.PuBuGn_9.mpl_colormap
cmap = palettable.colorbrewer.sequential.PuBu_9.mpl_colormap

cmap = palettable.colorbrewer.diverging.PRGn_4.mpl_colormap

cmap = palettable.scientific.sequential.Bamako_8.mpl_colormap
cmap = palettable.scientific.sequential.Bilbao_15.mpl_colormap
# cmap = palettable.scientific.sequential.Oslo_18.mpl_colormap

# colors = bmap.mpl_colors
params = {
    'axes.labelsize': 26,
    'font.size': 26,
    'legend.fontsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': True,
    'figure.figsize': [10, 4]
}
plt.rcParams.update(params)

# grid(axis='y', color="0.9", linestyle='-', linewidth=1)
# frame.set_facecolor('1.0')
# frame.set_edgecolor('1.0')
save_path = "/Users/aidanscannell/Documents/phd-genral/conference-papers/neurips-2020-submission/images/model/"

filename = "./saved_models/18-3/1637/model"
saver = gpflow.saver.Saver()
with tf.Graph().as_default() as graph, tf.Session().as_default():
    m = saver.load(filename)
    m.compile()

    X = m.X.value
    Y = m.Y.value
    xy, xx, yy = create_grid(X, N=961)

    a_mu, a_var = m.predict_a(xy)  # Predict alpha values at test locations
    y_mu, y_var = m.predict_y(xy)  # Predict alpha values at test locations
    h_mu, h_var = m.predict_h(xy)  # Predict alpha values at test locations
    f_mus, f_vars = m.predict_f(xy)  # Predict alpha values at test locations

bounds = np.linspace(0, 1, 11)
fig, ax = plot_contourf(
    xx,
    yy,
    a_mu.reshape(xx.shape),
    clabel=False,
    cbar_label='$P(\\alpha_*=0 | \mathbf{x}_*, \mathcal{D}, \mathbf{\\theta})$',
    levels=bounds)
plt.savefig(save_path + 'alpha.pdf', transparent=True)

data = np.load('../data/npz/turbulence/model_data_fan_fixed_subset.npz')
X = data['x']
Y = data['y'][:, 0:1]
fig, ax = plot_contourf(X[:, 0],
                        X[:, 1],
                        Y.reshape(-1),
                        tri=True,
                        clabel=False,
                        cbar_label='$\Delta x$')
plt.savefig(save_path + 'dataset.pdf', transparent=True)
ax.scatter(X[:, 0], X[:, 1], marker='x', color='k', alpha=0.7, s=0.5)
num = 380
line = ax.plot(X[0:num, 0], X[0:num, 1], color='k', alpha=0.5)[0]
add_arrow(line, color='k', alpha=0.5)
plt.savefig(save_path + 'dataset-arrow.pdf', transparent=True)

params = {
    'axes.labelsize': 32,
    'font.size': 32,
    'legend.fontsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'text.usetex': True,
    'figure.figsize': [20, 7]
}
plt.rcParams.update(params)

fig, ax = plot_mean_and_var(xx,
                            yy,
                            y_mu.reshape(xx.shape),
                            y_var.reshape(xx.shape),
                            clabel=False,
                            left_label='$\\mu_{y_*}$',
                            right_label='diag($\\Sigma_{y_*}$)')
plt.savefig(save_path + 'y_dim_1.pdf', transparent=True)

fig, ax = plot_mean_and_var(xx,
                            yy,
                            h_mu.reshape(xx.shape),
                            h_var.reshape(xx.shape),
                            clabel=False,
                            left_label='$\\mu_{h_*}$',
                            right_label='diag($\\Sigma_{h_*}$)')
plt.savefig(save_path + 'h.pdf', transparent=True)

for i, (f_mu, f_var) in enumerate(zip(f_mus, f_vars)):

    fig, ax = plot_mean_and_var(xx,
                                yy,
                                f_mu[:, 0].reshape(xx.shape),
                                f_var[:, 0].reshape(xx.shape),
                                clabel=False,
                                left_label='$\\mu_{f_' + str(i + 1) + '}$',
                                right_label='diag($\\Sigma_{f_' + str(i + 1) +
                                '}$)')
    plt.savefig(save_path + 'f' + str(i + 1) + '_dim_1.pdf', transparent=True)

plt.show()
