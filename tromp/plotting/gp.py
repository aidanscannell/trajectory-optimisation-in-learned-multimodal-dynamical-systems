import matplotlib.pyplot as plt
from matplotlib import cm
from tromp.plotting.utils import create_grid
import jax.numpy as jnp

cmap = cm.coolwarm
cmap = cm.PRGn
cmap = cm.PiYG


def plot_contourf(fig, ax, x, y, z, label="", levels=20, cbar=True):
    contf = ax.contourf(
        x,
        y,
        z.reshape(x.shape),
        cmap=cmap,
        levels=levels,
        antialiased=False,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if cbar:
        cbar = fig.colorbar(contf, shrink=0.5, aspect=5, ax=ax)
        cbar.set_label(label)


def plot_tricontourf(fig, ax, x, y, z, label=""):
    cont = ax.tricontourf(x, y, z, 15)
    cbar = fig.colorbar(cont, shrink=0.5, aspect=5, ax=ax)
    cbar.set_label(label)


def plot_mean_and_var(xx, yy, mu, var, llabel="mean", rlabel="variance"):
    # fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
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


############################################################
# Methods for plotting from an instance of gpjax.models.svgp
############################################################


def plot_svgp_mean_and_var(svgp, mean_label="Mean", var_label="Variance"):
    Xnew, xx, yy = create_grid(svgp.inducing_variable, 961)

    # hack for icra plots - should be removed
    N = 1000
    sqrtN = int(jnp.sqrt(N))
    x1_low = -3
    x1_high = 3
    x2_low = -3
    x2_high = 3
    xx = jnp.linspace(x1_low, x1_high, sqrtN)
    yy = jnp.linspace(x2_low, x2_high, sqrtN)
    xx, yy = jnp.meshgrid(xx, yy)
    Xnew = jnp.column_stack([xx.reshape(-1), yy.reshape(-1)])

    fmean, fvar = svgp.predict_f(Xnew, full_cov=False)
    return plot_mean_and_var(xx, yy, fmean, fvar, llabel=mean_label, rlabel=var_label)


def plot_svgp_jacobian_mean(svgp):
    Xnew, xx, yy = create_grid(svgp.inducing_variable, 961)
    fmean, fvar = svgp.predict_f(Xnew, full_cov=False)
    jac_mean, jac_var = svgp.predict_jacobian_f_wrt_Xnew(Xnew, full_cov=False)
    return plot_jacobian_mean(xx, yy, Xnew, jac_mean, fmean, fvar)


def plot_svgp_jacobian_var(svgp):
    Xnew, xx, yy = create_grid(svgp.inducing_variable, 961)
    _, jac_var = svgp.predict_jacobian_f_wrt_Xnew(Xnew, full_cov=True)
    print("jac_var")
    print(jac_var.shape)
    return plot_jacobian_var(xx, yy, Xnew, jac_var)
