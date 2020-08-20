import jax.numpy as np
import matplotlib.pyplot as plt

from jax import vmap, partial, jit

from ProbGeo.gp import gp_jacobian, gp_jacobian_hard_coded
from ProbGeo.typing import InputData, OutputData, MeanFunc, MeanAndVariance


def gp_metric_tensor(Xnew: InputData,
                     X: InputData,
                     kernel,
                     mean_func: MeanFunc,
                     Y: OutputData,
                     full_cov: bool = True,
                     cov_weight: np.float64 = 1.,
                     jitter=1.e-4):
    def calc_expected_metric_tensor_single(x):
        if len(x.shape) == 1:
            x = x.reshape(1, input_dim)

        mu_j, cov_j = gp_jacobian(
            # Xnew,
            x,
            X,
            kernel,
            mean_func,
            Y,
            full_cov=full_cov,
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
    output_dim = Y.shape[1]

    expected_metric_tensor, mu_j, cov_j = vmap(
        calc_expected_metric_tensor_single, in_axes=(0))(Xnew)
    return expected_metric_tensor, mu_j, cov_j


# def gp_metric_tensor(test_inputs,
#                      cov_fn,
#                      X,
#                      Y,
#                      cov_weight,
#                      jitter=1.e-4,
#                      full_cov=True):
#     def calc_expected_metric_tensor_single(x):
#         if len(x.shape) == 1:
#             x = x.reshape(1, input_dim)
#         # mu_j, cov_j = gp_jacobian(cov_fn, x, X, Y, jitter=jitter)

#         mu_j, cov_j = gp_jacobian(
#             Xnew,
#             X,
#             kernel,
#             mean_func,
#             Y,
#             full_cov=full_cov,
#             # jitter=1e-8,
#             white=True)
#         # mu_j, cov_j = gp_jacobian_hard_coded(cov_fn, x, X, Y, jitter=jitter)
#         assert mu_j.shape == (input_dim, 1)
#         assert cov_j.shape == (input_dim, input_dim)

#         if not full_cov:
#             # TODO this should be inside gp_jacobian to improve comp speed
#             var_j = np.diagonal(cov_j)
#             cov_j = np.diag(var_j)

#         expected_jac_outer_prod = np.matmul(mu_j,
#                                             mu_j.T)  # [input_dim x input_dim]
#         assert expected_jac_outer_prod.shape == (input_dim, input_dim)

#         expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
#         assert expected_metric_tensor.shape == (input_dim, input_dim)
#         return expected_metric_tensor, mu_j, cov_j

#     num_test_inputs = test_inputs.shape[0]
#     input_dim = test_inputs.shape[1]
#     output_dim = Y.shape[1]

#     expected_metric_tensor, mu_j, cov_j = vmap(
#         calc_expected_metric_tensor_single, in_axes=(0))(test_inputs)
#     return expected_metric_tensor, mu_j, cov_j


@partial(jit, static_argnums=(1, 2))
def calc_vec_metric_tensor(pos, metric_fn, metric_fn_args):
    pos = pos.reshape(1, -1)
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

    X, Y, kernel = load_data_and_init_kernel_fake(
        filename='../models/saved_models/params_fake.npz')
    mean_func = 0.
    # cov_weight = 38
    cov_weight = 1.
    # cov_weight = 0.15

    # set the covariance function to use for the jacobian/metric
    # cov_fn = kernel.K

    # plot original GP
    Xnew, xx, yy = create_grid(X, N=961)
    mu, var = gp_predict(Xnew,
                         X,
                         kernel,
                         mean_func=mean_func,
                         Y=Y,
                         full_cov=False)
    # mu, cov = gp_predict(test_inputs, X, Y kernel)
    # var = np.diag(cov).reshape(-1, 1)
    axs = plot_mean_and_var(xx, yy, mu, var)

    # calculate metric tensor and jacobian
    metric_tensor, mu_j, cov_j = gp_metric_tensor(Xnew,
                                                  X,
                                                  kernel,
                                                  mean_func=mean_func,
                                                  Y=Y,
                                                  full_cov=True,
                                                  cov_weight=cov_weight)
    # metric_tensor, mu_j, cov_j = gp_metric_tensor(Xnew,
    #                                               cov_fn,
    #                                               X,
    #                                               Y,
    #                                               cov_weight=cov_weight)

    axs = plot_jacobian_mean(xx, yy, Xnew, mu_j, mu, var)
    plt.suptitle('E(J)')

    axs = plot_jacobian_var(xx, yy, Xnew, cov_j)
    plt.suptitle('Cov(J)')

    axs = plot_metric_trace(xx, yy, metric_tensor)

    axs = plot_scatter_matrix(Xnew, metric_tensor)
    plt.suptitle('G(x)')

    plt.show()
