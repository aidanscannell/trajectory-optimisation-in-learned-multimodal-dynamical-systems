import jax.numpy as np
import jax.scipy as sp
import matplotlib.pyplot as plt

from jax import jacrev, jacfwd, vmap, partial, jit
from jax.config import config
config.update("jax_enable_x64", True)

from ProbGeo.svgp import svgp_jacobian
from ProbGeo.typing import InputData, OutputData, MeanFunc, MeanAndVariance


def svgp_metric_tensor(Xnew: InputData,
                       Z: InputData,
                       kernel,
                       mean_func: MeanFunc,
                       q_mu: OutputData,
                       full_cov: bool = True,
                       q_sqrt=None,
                       cov_weight: np.float64 = 1.,
                       jitter: np.float64 = 1.e-4):
    def calc_expected_metric_tensor_single(x):
        if len(x.shape) == 1:
            x = x.reshape(1, input_dim)
        mu_j, cov_j = svgp_jacobian(x,
                                    Z,
                                    kernel,
                                    mean_func,
                                    q_mu,
                                    full_cov=full_cov,
                                    q_sqrt=q_sqrt,
                                    jitter=jitter,
                                    white=True)
        mu_j = mu_j.reshape(input_dim, 1)
        cov_j = cov_j.reshape(input_dim, input_dim)
        # TODO move this reshape to inside svgp_jac
        assert mu_j.shape == (input_dim, 1)
        assert cov_j.shape == (input_dim, input_dim)

        expected_jac_outer_prod = np.matmul(mu_j,
                                            mu_j.T)  # [input_dim x input_dim]
        assert expected_jac_outer_prod.shape == (input_dim, input_dim)

        expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
        assert expected_metric_tensor.shape == (input_dim, input_dim)
        return expected_metric_tensor, mu_j, cov_j

    num_test_inputs = Xnew.shape[0]
    input_dim = Xnew.shape[1]
    output_dim = q_mu.shape[1]

    expected_metric_tensor, mu_j, cov_j = vmap(
        calc_expected_metric_tensor_single, in_axes=(0))(Xnew)
    return expected_metric_tensor, mu_j, cov_j


@partial(jit, static_argnums=(1, 2))
def calc_vec_metric_tensor(pos, metric_fn, metric_fn_args):
    pos = pos.reshape(1, -1)
    metric_tensor, _, _ = metric_fn(pos, *metric_fn_args)
    input_dim = pos.shape[1]
    vec_metric_tensor = metric_tensor.reshape(input_dim * input_dim, )
    return vec_metric_tensor


if __name__ == "__main__":
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse
    from ProbGeo.visualisation.gp import plot_mean_and_var, plot_jacobian_mean, plot_jacobian_var
    from ProbGeo.visualisation.metric import plot_scatter_matrix, plot_metric_trace
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.svgp import svgp_predict
    jitter = 1e-4

    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../models/saved_models/params_from_model.npz')
    cov_weight = 50

    # plot original GP
    Xnew, xx, yy = create_grid(X, N=961)
    mu, var = svgp_predict(
        Xnew,
        Z,
        kernel,
        mean_func,
        q_mu,
        full_cov=False,
        # full_cov=True,
        q_sqrt=q_sqrt,
        jitter=jitter)
    axs = plot_mean_and_var(xx, yy, mu, var)

    metric_tensor, mu_j, cov_j = svgp_metric_tensor(Xnew,
                                                    Z,
                                                    kernel,
                                                    mean_func,
                                                    q_mu,
                                                    full_cov=False,
                                                    q_sqrt=q_sqrt,
                                                    cov_weight=cov_weight,
                                                    jitter=jitter)

    axs = plot_jacobian_mean(xx, yy, Xnew, mu_j, mu, var)
    plt.suptitle('E(J)')

    axs = plot_jacobian_var(xx, yy, Xnew, cov_j)
    plt.suptitle('Cov(J)')

    axs = plot_metric_trace(xx, yy, metric_tensor)

    axs = plot_scatter_matrix(Xnew, metric_tensor)
    plt.suptitle('G(x)')

    plt.show()
