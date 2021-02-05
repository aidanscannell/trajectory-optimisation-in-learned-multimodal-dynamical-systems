import jax.numpy as jnp
import matplotlib.pyplot as plt
from gpjax.custom_types import InputData
from tromp.metric_tensors import MetricTensor
from tromp.visualisation.gp import plot_contourf, plot_tricontourf
from tromp.visualisation.utils import create_grid


def plot_svgp_metric_trace(metric_tensor: MetricTensor, num_test: int = 961):
    Xnew, xx, yy = create_grid(metric_tensor.gp.inducing_variable, num_test)
    tensor = metric_tensor.metric_fn(Xnew)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    trace_metric_tensor = jnp.trace(tensor, axis1=1, axis2=2)
    plot_contourf(fig, ax, xx, yy, trace_metric_tensor)
    plt.title("Tr(G(x))")
    return fig, ax


# def plot_scatter_matrix(X, scatter2x2, labels=["", "", "", ""]):
#     fig, axs = plt.subplots(2, 2, figsize=(24, 8))
#     plt.subplots_adjust(wspace=0, hspace=0)

#     plot_tricontourf(
#         fig, axs[0, 0], X[:, 0], X[:, 1], scatter2x2[:, 0, 0], label=labels[0]
#     )
#     plot_tricontourf(
#         fig, axs[0, 1], X[:, 0], X[:, 1], scatter2x2[:, 0, 1], label=labels[1]
#     )
#     plot_tricontourf(
#         fig, axs[1, 0], X[:, 0], X[:, 1], scatter2x2[:, 1, 0], label=labels[2]
#     )
#     plot_tricontourf(
#         fig, axs[1, 1], X[:, 0], X[:, 1], scatter2x2[:, 1, 1], label=labels[3]
#     )
#     return axs
