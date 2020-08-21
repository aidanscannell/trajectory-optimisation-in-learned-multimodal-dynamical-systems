import jax.numpy as np
import matplotlib.pyplot as plt

from jax import vmap, partial, jit

from ProbGeo.gp import gp_jacobian, gp_jacobian_hard_coded
from ProbGeo.typing import InputData, OutputData, MeanFunc, MeanAndVariance


def gp_metric_tensor(Xnew: InputData,
                     X: InputData,
                     kernel,
                     mean_func: MeanFunc,
                     f: OutputData,
                     full_cov: bool = True,
                     q_sqrt=None,
                     cov_weight: np.float64 = 1.,
                     jitter=1.e-4,
                     white: bool = True):
    def calc_expected_metric_tensor_single(x):
        if len(x.shape) == 1:
            x = x.reshape(1, input_dim)

        mu_j, cov_j = gp_jacobian(
            # Xnew,
            x,
            X,
            kernel,
            mean_func=mean_func,
            f=f,
            full_cov=full_cov,
            q_sqrt=q_sqrt,
            jitter=jitter,
            white=True)
        # mu_j, cov_j = gp_jacobian_hard_coded(cov_fn, x, X, Y, jitter=jitter)
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
    output_dim = f.shape[1]

    expected_metric_tensor, mu_j, cov_j = vmap(
        calc_expected_metric_tensor_single, in_axes=(0))(Xnew)
    return expected_metric_tensor, mu_j, cov_j


@partial(jit, static_argnums=(1, 2))
def calc_vec_metric_tensor(pos, metric_fn, metric_fn_kwargs):
    pos = pos.reshape(1, -1)
    metric_tensor, _, _ = metric_fn(pos, **metric_fn_kwargs)
    input_dim = pos.shape[1]
    vec_metric_tensor = metric_tensor.reshape(input_dim * input_dim, )
    return vec_metric_tensor


def test_metric_tensor():
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake
    from ProbGeo.visualisation.gp import plot_mean_and_var, plot_jacobian_mean, plot_jacobian_var
    from ProbGeo.visualisation.metric import plot_scatter_matrix, plot_metric_trace
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict

    X, Y, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    mean_func = 0.
    cov_weight = 1.

    # plot original GP
    Xnew, xx, yy = create_grid(X, N=961)
    mu, var = gp_predict(Xnew,
                         X,
                         kernel,
                         mean_func=mean_func,
                         f=Y,
                         full_cov=False)
    axs = plot_mean_and_var(xx, yy, mu, var)

    # calculate metric tensor and jacobian
    metric_tensor, mu_j, cov_j = gp_metric_tensor(Xnew,
                                                  X,
                                                  kernel,
                                                  mean_func=mean_func,
                                                  f=Y,
                                                  full_cov=True,
                                                  cov_weight=cov_weight)

    axs = plot_jacobian_mean(xx, yy, Xnew, mu_j, mu, var)
    plt.suptitle('E(J)')

    axs = plot_jacobian_var(xx, yy, Xnew, cov_j)
    plt.suptitle('Cov(J)')

    axs = plot_metric_trace(xx, yy, metric_tensor)

    axs = plot_scatter_matrix(Xnew, metric_tensor)
    plt.suptitle('G(x)')

    plt.show()


def test_metric_tensor_with_svgp():
    from ProbGeo.utils.gp import load_data_and_init_kernel_sparse
    from ProbGeo.visualisation.gp import plot_mean_and_var, plot_jacobian_mean, plot_jacobian_var
    from ProbGeo.visualisation.metric import plot_scatter_matrix, plot_metric_trace
    from ProbGeo.visualisation.utils import create_grid
    from ProbGeo.gp import gp_predict

    X, Z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel_sparse(
        filename='../models/saved_models/params_fake_sparse_20-08_2.npz')
    # filename='../models/saved_models/params_from_model.npz')
    cov_weight = 50
    cov_weight = 1

    # plot original GP
    Xnew, xx, yy = create_grid(X, N=961)
    mu, var = gp_predict(
        Xnew,
        Z,
        kernel,
        mean_func=mean_func,
        f=q_mu,
        full_cov=True,
        # full_cov=False,
        # white=False,
        q_sqrt=q_sqrt)
    var = np.diag(var)
    print('here')
    print(var.shape)

    axs = plot_mean_and_var(xx, yy, mu, var)

    # calculate metric tensor and jacobian
    metric_tensor, mu_j, cov_j = gp_metric_tensor(Xnew,
                                                  Z,
                                                  kernel,
                                                  mean_func=mean_func,
                                                  f=q_mu,
                                                  full_cov=True,
                                                  q_sqrt=q_sqrt,
                                                  cov_weight=cov_weight)

    axs = plot_jacobian_mean(xx, yy, Xnew, mu_j, mu, var)
    plt.suptitle('E(J)')

    axs = plot_jacobian_var(xx, yy, Xnew, cov_j)
    plt.suptitle('Cov(J)')

    axs = plot_metric_trace(xx, yy, metric_tensor)

    axs = plot_scatter_matrix(Xnew, metric_tensor)
    plt.suptitle('G(x)')

    plt.show()


if __name__ == "__main__":

    # test_metric_tensor()
    test_metric_tensor_with_svgp()
