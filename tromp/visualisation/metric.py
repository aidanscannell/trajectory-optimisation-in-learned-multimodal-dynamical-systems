import matplotlib.pyplot as plt
import numpy as np
from ProbGeo.visualisation.gp import plot_mean_and_var, plot_contourf, plot_tricontourf


def plot_metric_trace(x, y, metric_tensor):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    trace_metric_tensor = np.trace(metric_tensor, axis1=1, axis2=2)
    plot_contourf(fig, ax, x, y, trace_metric_tensor)
    plt.title('Tr(G(x))')
    return fig, ax


def plot_scatter_matrix(X, scatter2x2, labels=["", "", "", ""]):
    fig, axs = plt.subplots(2, 2, figsize=(24, 8))
    plt.subplots_adjust(wspace=0, hspace=0)

    plot_tricontourf(fig,
                     axs[0, 0],
                     X[:, 0],
                     X[:, 1],
                     scatter2x2[:, 0, 0],
                     label=labels[0])
    plot_tricontourf(fig,
                     axs[0, 1],
                     X[:, 0],
                     X[:, 1],
                     scatter2x2[:, 0, 1],
                     label=labels[1])
    plot_tricontourf(fig,
                     axs[1, 0],
                     X[:, 0],
                     X[:, 1],
                     scatter2x2[:, 1, 0],
                     label=labels[2])
    plot_tricontourf(fig,
                     axs[1, 1],
                     X[:, 0],
                     X[:, 1],
                     scatter2x2[:, 1, 1],
                     label=labels[3])
    return axs
