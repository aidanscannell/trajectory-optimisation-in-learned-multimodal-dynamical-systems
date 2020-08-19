import jax.numpy as np
import jax.scipy as sp
import matplotlib.pyplot as plt

from jax import jacrev, jacfwd, vmap, partial, jit
from jax.config import config
config.update("jax_enable_x64", True)

from ProbGeo.svgp import sparse_gp_jacobian


def gp_metric_tensor(test_inputs,
                     cov_fn,
                     X,
                     Y,
                     cov_weight,
                     jitter=1.e-4,
                     full_cov=True):
    def calc_expected_metric_tensor_single(x):
        if len(x.shape) == 1:
            x = x.reshape(1, input_dim)
        mu_j, cov_j = gp_jacobian(cov_fn, x, X, Y, jitter=jitter)
        # mu_j, cov_j = gp_jacobian_hard_coded(cov_fn, x, X, Y, jitter=jitter)
        assert mu_j.shape == (input_dim, 1)
        assert cov_j.shape == (input_dim, input_dim)

        if not full_cov:
            # TODO this should be inside gp_jacobian to improve comp speed
            var_j = np.diagonal(cov_j)
            cov_j = np.diag(var_j)

        expected_jac_outer_prod = np.matmul(mu_j,
                                            mu_j.T)  # [input_dim x input_dim]
        assert expected_jac_outer_prod.shape == (input_dim, input_dim)

        expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
        assert expected_metric_tensor.shape == (input_dim, input_dim)
        return expected_metric_tensor, mu_j, cov_j

    num_test_inputs = test_inputs.shape[0]
    input_dim = test_inputs.shape[1]
    output_dim = Y.shape[1]

    expected_metric_tensor, mu_j, cov_j = vmap(
        calc_expected_metric_tensor_single, in_axes=(0))(test_inputs)
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
    from ProbGeo.svgp import sparse_gp_predict

    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../models/saved_models/params_from_model.npz')
    # filename='../models/saved_models/params_fake_sparse.npz')
    cov_weight = 38
    cov_weight = 0.15
    print(X.shape)
    print(Z.shape)
    print(q_mu.shape)
    print(q_sqrt.shape)
    print(mean_func.shape)

    # set the covariance function to use for the jacobian/metric
    # cov_fn = kernel.K

    # plot original GP
    Xnew, xx, yy = create_grid(X, N=961)
    mu, cov = sparse_gp_predict(
        Xnew,
        Z,
        kernel,
        mean_func,
        q_mu,
        full_cov=False,
        # full_cov=True,
        q_sqrt=q_sqrt,
        jitter=1e-6)
    # mu, cov = gp_predict(test_inputs, X, a_mu, kernel)
    # var = np.diag(cov).reshape(-1, 1)
    var = cov
    axs = plot_mean_and_var(xx, yy, mu, var)
    # plt.show()

    mu_j, cov_j = sparse_gp_jacobian(Xnew,
                                     Z,
                                     kernel,
                                     mean_func,
                                     q_mu,
                                     full_cov=True,
                                     q_sqrt=q_sqrt)
    # # calculate metric tensor and jacobian
    # metric_tensor, mu_j, cov_j = gp_metric_tensor(test_inputs,
    #                                               cov_fn,
    #                                               X,
    #                                               Y,
    #                                               cov_weight=cov_weight)

    axs = plot_jacobian_mean(xx, yy, Xnew, mu_j, mu, var)
    plt.suptitle('E(J)')

    axs = plot_jacobian_var(xx, yy, Xnew, cov_j)
    plt.suptitle('Cov(J)')
    plt.show()

    # axs = plot_metric_trace(xx, yy, metric_tensor)

    # axs = plot_scatter_matrix(test_inputs, metric_tensor)
    # plt.suptitle('G(x)')

    # plt.show()
