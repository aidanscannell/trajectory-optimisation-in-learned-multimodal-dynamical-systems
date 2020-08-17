import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm


def plot_contourf(fig, ax, x, y, z, label=''):
    contf = ax.contourf(x,
                        y,
                        z.reshape(x.shape),
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False)
    cbar = fig.colorbar(contf, shrink=0.5, aspect=5, ax=ax)
    cbar.set_label(label)


def plot_tricontourf(fig, ax, x, y, z, label=''):
    cont = ax.tricontourf(x, y, z, 15)
    cbar = fig.colorbar(cont, shrink=0.5, aspect=5, ax=ax)
    cbar.set_label(label)


def plot_mean_and_var(xx, yy, mu, var, llabel='mean', rlabel='variance'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # plt.subplots_adjust(wspace=0, hspace=0)
    plot_contourf(fig, axs[0], xx, yy, mu, label=llabel)
    plot_contourf(fig, axs[1], xx, yy, var, label=rlabel)
    return fig, axs


def plot_jacobian_mean(xx, yy, xy, mu_j, mu, var):
    fig, axs = plot_mean_and_var(xx, yy, mu, var)
    for ax in axs:
        ax.quiver(xy[:, 0], xy[:, 1], mu_j[:, 0, 0], mu_j[:, 1, 0])
    return fig, axs


def plot_jacobian_var(xx, yy, xy, cov_j):
    fig, axs = plt.subplots(2, 2, figsize=(24, 8))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            plot_contourf(fig, axs[i, j], xx, yy, cov_j[:, i, j])
    return fig, axs
