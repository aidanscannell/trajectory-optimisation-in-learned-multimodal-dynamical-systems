import jax.numpy as np
import jax.scipy as sp
import matplotlib.pyplot as plt

from jax import jacrev, jacfwd, vmap, partial, jit


def grad_cov_fn_wrt_x1(cov_fn, x1, x2):
    """Calculate derivative of cov_fn wrt to x1

    :param cov_fn: covariance function with signature cov_fn(x1, x2)
    :param x1: [1, input_dim]
    :param x2: [num_x2, input_dim]
    :returns:
    """
    dk = jacfwd(cov_fn, (0))(x1, x2)
    # TODO replace squeeze with correct dimensions
    dk = np.squeeze(dk)
    return dk


def grad_cov_fn_wrt_x1x2(cov_fn, x1, x2):
    d2k = jacrev(jacfwd(cov_fn, (1)), (0))(x1, x2)
    # TODO replace squeeze with correct dimensions
    d2k = np.squeeze(d2k)
    return d2k


def gp_jacobian(cov_fn, Xnew, X, Y, jitter=1e-4):
    assert Xnew.shape[1] == X.shape[1]
    Kxx = cov_fn(X, X)
    Kxx += jitter * np.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    dKdx1 = grad_cov_fn_wrt_x1(cov_fn, Xnew, X)
    d2K = grad_cov_fn_wrt_x1x2(cov_fn, Xnew, Xnew)

    A1 = sp.linalg.solve_triangular(chol, dKdx1, lower=True)
    A2 = sp.linalg.solve_triangular(chol, Y, lower=True)
    ATA = A1.T @ A1

    mu_j = A1.T @ A2
    cov_j = d2K - ATA
    # cov_j = 7 + d2K - ATA
    cov_j = 7 - ATA
    return mu_j, cov_j


def gp_metric_tensor(test_inputs, cov_fn, X, Y, cov_weight, jitter=1.e-4):
    # def gp_metric_tensor(test_inputs, cov_fn, X, Y, cov_weight, jitter=1.e-4):
    def calc_expected_metric_tensor_single(x):
        if len(x.shape) == 1:
            x = x.reshape(1, input_dim)
        mu_j, cov_j = gp_jacobian(cov_fn, x, X, Y, jitter=jitter)
        assert mu_j.shape == (input_dim, 1)
        assert cov_j.shape == (input_dim, input_dim)

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
    metric_tensor, _, _ = metric_fn(pos, *metric_fn_args)
    input_dim = pos.shape[1]
    vec_metric_tensor = metric_tensor.reshape(input_dim * input_dim, )
    return vec_metric_tensor


if __name__ == "__main__":
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    from ProbGeo.visualisation.gp import plot_mean_and_var, plot_jacobian_mean, plot_jacobian_var
    from ProbGeo.visualisation.metric import plot_scatter_matrix, plot_metric_trace
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict

    X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    Y = a_mu
    cov_weight = 38
    cov_weight = 0.15

    # set the covariance function to use for the jacobian/metric
    cov_fn = kernel.K

    # plot original GP
    test_inputs, xx, yy = create_grid(X, N=961)
    mu, cov = gp_predict(test_inputs, X, a_mu, kernel)
    var = np.diag(cov).reshape(-1, 1)
    axs = plot_mean_and_var(xx, yy, mu, var)

    # calculate metric tensor and jacobian
    metric_tensor, mu_j, cov_j = gp_metric_tensor(test_inputs,
                                                  cov_fn,
                                                  X,
                                                  Y,
                                                  cov_weight=cov_weight)

    axs = plot_jacobian_mean(xx, yy, test_inputs, mu_j, mu, var)
    plt.suptitle('E(J)')

    axs = plot_jacobian_var(xx, yy, test_inputs, cov_j)
    plt.suptitle('Cov(J)')

    axs = plot_metric_trace(xx, yy, metric_tensor)

    axs = plot_scatter_matrix(test_inputs, metric_tensor)
    plt.suptitle('G(x)')

    plt.show()
