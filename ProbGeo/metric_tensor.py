import jax
import jax.numpy as np
from gpjax.custom_types import InputData, MeanFunc, OutputData
from gpjax.kernels import Kernel
from gpjax.prediction import gp_jacobian


# @jax.partial(jax.jit, static_argnums=(1, 2))
def metric_tensor_fn(Xnew, fun, fun_kwargs):
    def single_jac(x_new):
        # x_new = x_new.reshape(1, -1)
        jac = jax.jacfwd(fun, (0))(x_new, **fun_kwargs)
        print("single jac")
        print(jac.shape)
        return jac

    input_dim = Xnew.shape[1]
    num_test = Xnew.shape[0]

    jac = jax.vmap(single_jac)(Xnew)

    print('og jac')
    print(jac.shape)

    metric_tensor = np.matmul(np.transpose(jac, (0, 2, 1)),
                              jac)  # [input_dim x input_dim]
    assert metric_tensor.shape == (num_test, input_dim, input_dim)
    print('metric yo')
    print(metric_tensor.shape)
    # print(metric_tensor)

    # return metric_tensor.reshape(input_dim, input_dim), jac
    return metric_tensor, jac


# def mogpe_mixing_prob_metric_tensor(Xnew: InputData,
#                                     X: InputData,
#                                     kernel,
#                                     mean_func: MeanFunc,
#                                     f: OutputData,
#                                     full_cov: bool = True,
#                                     q_sqrt=None,
#                                     cov_weight: np.float64 = 1.,
#                                     jitter=1.e-4,
#                                     white: bool = True):
#     jac = mogpe_mixing_prob_jacobian(Xnew, X, kernel, mean_func, f, full_cov,
#                                      q_sqrt, jitter, white)
#     input_dim = Xnew.shape[1]
#     num_test = Xnew.shape[0]

#     metric_tensor = np.matmul(jac, np.transpose(
#         jac, (0, 2, 1)))  # [input_dim x input_dim]
#     assert metric_tensor.shape == (num_test, input_dim, input_dim)

#     return metric_tensor, jac


# @jax.partial(jax.jit, static_argnums=(1, 2, 3, 4))
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
            mean_func,
            f=f,
            full_cov=full_cov,
            # full_cov=False,
            q_sqrt=q_sqrt,
            jitter=jitter,
            white=True)
        # mu_j, cov_j = gp_jacobian_hard_coded(cov_fn, x, X, Y, jitter=jitter)
        assert mu_j.shape == (input_dim, 1)
        assert cov_j.shape == (input_dim, input_dim)

        expected_jac_outer_prod = np.matmul(mu_j,
                                            mu_j.T)  # [input_dim x input_dim]
        assert expected_jac_outer_prod.shape == (input_dim, input_dim)

        # expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
        expected_metric_tensor = expected_jac_outer_prod + cov_weight * cov_j
        # expected_metric_tensor = cov_weight * cov_j.T
        # expected_metric_tensor = cov_j
        assert expected_metric_tensor.shape == (input_dim, input_dim)
        return expected_metric_tensor, mu_j, cov_j

    num_test_inputs = Xnew.shape[0]
    input_dim = Xnew.shape[1]
    output_dim = f.shape[1]
    print('inside gp_metric_tensor')
    print(output_dim)

    expected_metric_tensor, mu_j, cov_j = jax.vmap(
        calc_expected_metric_tensor_single, in_axes=(0))(Xnew)
    return expected_metric_tensor, mu_j, cov_j


@jax.partial(jax.jit, static_argnums=(1, 2))
def calc_vec_metric_tensor(pos, metric_fn, metric_fn_kwargs):
    print('here calc vec metric tensor')
    print(pos.shape)
    pos = pos.reshape(1, -1)
    try:
        metric_tensor, _ = metric_fn(pos, **metric_fn_kwargs)
    except:
        metric_tensor, _, _ = metric_fn(pos, **metric_fn_kwargs)
    input_dim = pos.shape[1]
    vec_metric_tensor = metric_tensor.reshape(input_dim * input_dim, )
    return vec_metric_tensor
