import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import objax
from gpjax.custom_types import InputData
# from gpjax.kernels import Kernel
from gpjax.models.gp import SVGP, GPBase
from gpjax.prediction import gp_jacobian
from gpjax.utilities import leading_transpose

MetricTensor = jnp.ndarray

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

    print("og jac")
    print(jac.shape)

    metric_tensor = jnp.matmul(
        jnp.transpose(jac, (0, 2, 1)), jac
    )  # [input_dim x input_dim]
    assert metric_tensor.shape == (num_test, input_dim, input_dim)
    print("metric yo")
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
#                                     cov_weight: jnp.float64 = 1.,
#                                     jitter=1.e-4,
#                                     white: bool = True):
#     jac = mogpe_mixing_prob_jacobian(Xnew, X, kernel, mean_func, f, full_cov,
#                                      q_sqrt, jitter, white)
#     input_dim = Xnew.shape[1]
#     num_test = Xnew.shape[0]

#     metric_tensor = jnp.matmul(jac, jnp.transpose(
#         jac, (0, 2, 1)))  # [input_dim x input_dim]
#     assert metric_tensor.shape == (num_test, input_dim, input_dim)

#     return metric_tensor, jac


# # @jax.partial(jax.jit, static_argnums=(1, 2, 3, 4))
# def gp_metric_tensor(
#     Xnew: InputData,
#     X: InputData,
#     kernel: Kernel,
#     mean_func: MeanFunc,
#     f: OutputData,
#     full_cov: bool = True,
#     q_sqrt=None,
#     cov_weight: jnp.float64 = 1.0,
#     jitter=1.0e-4,
#     white: bool = True,
# ):
#     def calc_expected_metric_tensor_single(x):
#         if len(x.shape) == 1:
#             x = x.reshape(1, input_dim)

#         mu_j, cov_j = gp_jacobian(
#             # Xnew,
#             x,
#             X,
#             kernel,
#             mean_func,
#             f=f,
#             full_cov=full_cov,
#             # full_cov=False,
#             q_sqrt=q_sqrt,
#             jitter=jitter,
#             white=True,
#         )
#         # mu_j, cov_j = gp_jacobian_hard_coded(cov_fn, x, X, Y, jitter=jitter)
#         assert mu_j.shape == (input_dim, 1)
#         assert cov_j.shape == (input_dim, input_dim)

#         expected_jac_outer_prod = jnp.matmul(
#             mu_j, mu_j.T
#         )  # [input_dim x input_dim]
#         assert expected_jac_outer_prod.shape == (input_dim, input_dim)

#         # expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
#         expected_metric_tensor = expected_jac_outer_prod + cov_weight * cov_j
#         # expected_metric_tensor = cov_weight * cov_j.T
#         # expected_metric_tensor = cov_j
#         assert expected_metric_tensor.shape == (input_dim, input_dim)
#         return expected_metric_tensor, mu_j, cov_j

#     num_test_inputs = Xnew.shape[0]
#     input_dim = Xnew.shape[1]
#     output_dim = f.shape[1]
#     print("inside gp_metric_tensor")
#     print(output_dim)

#     expected_metric_tensor, mu_j, cov_j = jax.vmap(
#         calc_expected_metric_tensor_single, in_axes=(0)
#     )(Xnew)
#     return expected_metric_tensor, mu_j, cov_j


# @jax.partial(jax.jit, static_argnums=(1, 2))
# def calc_vec_metric_tensor(pos, metric_fn, metric_fn_kwargs):
#     print("here calc vec metric tensor")
#     print(pos.shape)
#     pos = pos.reshape(1, -1)
#     try:
#         metric_tensor, _ = metric_fn(pos, **metric_fn_kwargs)
#         # TODO add the correct exception
#     except:
#         metric_tensor, _, _ = metric_fn(pos, **metric_fn_kwargs)
#     input_dim = pos.shape[1]
#     vec_metric_tensor = metric_tensor.reshape(
#         input_dim * input_dim,
#     )
#     return vec_metric_tensor


class MetricTensorBase(objax.Module):
    def metric_fn(self, Xnew: InputData) -> MetricTensor:
        raise NotImplementedError

    def __call__(self, Xnew: InputData) -> MetricTensor:
        return self.metric_fn(Xnew)


class RiemannianMetricTensor(MetricTensorBase):
    def metric_fn(self, Xnew) -> MetricTensor:
        raise NotImplementedError


class SVGPMetricTensor(RiemannianMetricTensor):
    def __init__(
        self,
        gp: SVGP,
        covariance_weight: jnp.float64 = 1.0,
        jitter: jnp.float64 = 1.0e-4,
    ):
        self.gp = gp
        self.covariance_weight = covariance_weight
        self.jitter = jitter

    def __call__(self, Xnew: InputData, full_cov: bool = True):
        return self.metric_fn(Xnew, full_cov=full_cov)

    def grad_vec_metric_tensor_wrt_Xnew(
        self, Xnew: InputData, full_cov: bool = True
    ):
        def calc_vec_metric_tensor(Xnew):
            # pos = pos.reshape(1, -1)
            print("inside vec metric tensor Xnew")
            print(Xnew.shape)
            metric_tensor = self.metric_fn(Xnew, full_cov=full_cov)
            # if full_cov = False:
            input_dim = Xnew.shape[-1]
            vec_metric_tensor = metric_tensor.reshape(
                input_dim * input_dim,
            )
            return vec_metric_tensor

        grad_func = jax.jacfwd(calc_vec_metric_tensor, 0)
        if len(Xnew.shape) == 1:
            grad_vec_metric_tensor_wrt_Xnew = grad_func(Xnew)
        else:
            grad_vec_metric_tensor_wrt_Xnew = jax.vmap(grad_func)(Xnew)
        return grad_vec_metric_tensor_wrt_Xnew

    # def calc_vec_metric_tensor(self, Xnew, full_cov: bool = True):
    #     # pos = pos.reshape(1, -1)
    #     print("inside vec metric tensor Xnew")
    #     print(Xnew.shape)
    #     metric_tensor = self.metric_fn(Xnew, full_cov=full_cov)
    #     input_dim = Xnew.shape[-1]
    #     vec_metric_tensor = metric_tensor.reshape(
    #         input_dim * input_dim,
    #     )
    #     return vec_metric_tensor

    def metric_fn(
        self,
        Xnew: InputData,
        full_cov: bool = True,
    ) -> MetricTensor:
        # jac_mean, jac_cov = gp_jacobian(
        #     Xnew,
        #     self.gp.inducing_variable,
        #     self.gp.kernel,
        #     self.gp.mean_function,
        #     f=self.gp.q_mu,
        #     full_cov=full_cov,
        #     q_sqrt=self.gp.q_sqrt,
        #     jitter=self.jitter,
        #     white=self.gp.whiten,
        # )
        print('inside metric_fn')
        print(Xnew.shape)
        input_dim = Xnew.shape[-1]
        if len(Xnew.shape) == 1:
            # Xnew = Xnew.reshape(1, -1)
            num_data = 1
        else:
            num_data = Xnew.shape[0]
        print(Xnew.shape)

        print("before gp.predict_jac what is Xnew.shape")
        print(Xnew.shape)
        jac_mean, jac_cov = self.gp.predict_jacobian_f_wrt_Xnew(
            Xnew, full_cov=full_cov
        )
        print("jac_mean all")
        print(jac_mean.shape)
        print(jac_cov.shape)
        print("num data")
        print(num_data)
        # TODO this is only true if full_cov=True
        # assert jac_mean.shape == (num_data, input_dim, 1)
        # assert jac_cov.shape == (num_data, input_dim, input_dim)

        jac_mean_trans = leading_transpose(jac_mean, [..., -1, -2])
        expected_jac_outer_prod = jnp.matmul(
            jac_mean, jac_mean_trans
        )  # [input_dim x input_dim]
        print("expected_jac_outer_prod")
        print(expected_jac_outer_prod.shape)
        # assert expected_jac_outer_prod.shape == (
        #     num_data,
        #     input_dim,
        #     input_dim,
        # )

        # expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
        expected_metric_tensor = (
            expected_jac_outer_prod + self.covariance_weight * jac_cov
        )
        print("expected_metric_tensor")
        print(expected_metric_tensor.shape)
        # expected_metric_tensor = cov_weight * cov_j.T
        # expected_metric_tensor = cov_j
        # assert expected_metric_tensor.shape == (num_data, input_dim, input_dim)
        return expected_metric_tensor

        # f = self.gp.q_mu
        # num_test_inputs = Xnew.shape[0]
        # input_dim = Xnew.shape[1]
        # output_dim = f.shape[1]

        # expected_metric_tensor, mu_j, cov_j = jax.vmap(
        #     calc_expected_metric_tensor_single, in_axes=(0)
        # )(Xnew)
        # return expected_metric_tensor, mu_j, cov_j
        # return expected_metric_tensor

    # def metric_fn(
    #     self,
    #     Xnew: InputData,
    #     full_cov: bool = True,
    # )-> MetricTensor:
    #     def calc_expected_metric_tensor_single(x):
    #         if len(x.shape) == 1:
    #             x = x.reshape(1, input_dim)

    #         mu_j, cov_j = gp_jacobian(
    #             x,
    #             self.gp.inducing_variable,
    #             self.gp.kernel,
    #             self.gp.mean_function,
    #             f=self.gp.q_mu,
    #             full_cov=full_cov,
    #             # full_cov=False,
    #             q_sqrt=self.gp.q_sqrt,
    #             jitter=self.jitter,
    #             white=self.gp.whiten,
    #         )
    #         # mu_j, cov_j = gp_jacobian_hard_coded(cov_fn, x, X, Y, jitter=jitter)
    #         assert mu_j.shape == (input_dim, 1)
    #         assert cov_j.shape == (input_dim, input_dim)

    #         expected_jac_outer_prod = jnp.matmul(
    #             mu_j, mu_j.T
    #         )  # [input_dim x input_dim]
    #         assert expected_jac_outer_prod.shape == (input_dim, input_dim)

    #         # expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
    #         expected_metric_tensor = (
    #             expected_jac_outer_prod + self.covariance_weight * cov_j
    #         )
    #         # expected_metric_tensor = cov_weight * cov_j.T
    #         # expected_metric_tensor = cov_j
    #         assert expected_metric_tensor.shape == (input_dim, input_dim)
    #         return expected_metric_tensor, mu_j, cov_j

    #     f = self.gp.q_mu
    #     num_test_inputs = Xnew.shape[0]
    #     input_dim = Xnew.shape[1]
    #     output_dim = f.shape[1]

    #     expected_metric_tensor, mu_j, cov_j = jax.vmap(
    #         calc_expected_metric_tensor_single, in_axes=(0)
    #     )(Xnew)
    #     # return expected_metric_tensor, mu_j, cov_j
    #     return expected_metric_tensor


# class GPMetricTensor(RiemannianMetricTensor):
#     def __init__(
#         self,
#         gp: GPBase,
#         covariance_weight: jnp.float64 = 1.0,
#         jitter: jnp.float64 = 1.0e-4,
#     ):
#         self.gp = gp
#         self.covariance_weight = covariance_weight
#         self.jitter = jitter

#     def metric_fn(
#         self,
#         Xnew: InputData,
#         full_cov: bool = True,
#     ):
#         def calc_expected_metric_tensor_single(x):
#             if len(x.shape) == 1:
#                 x = x.reshape(1, input_dim)

#             mu_j, cov_j = gp_jacobian(
#                 # Xnew,
#                 x,
#                 X,
#                 kernel,
#                 mean_func,
#                 f=f,
#                 full_cov=full_cov,
#                 # full_cov=False,
#                 q_sqrt=q_sqrt,
#                 jitter=jitter,
#                 white=True,
#             )
#             # mu_j, cov_j = gp_jacobian_hard_coded(cov_fn, x, X, Y, jitter=jitter)
#             assert mu_j.shape == (input_dim, 1)
#             assert cov_j.shape == (input_dim, input_dim)

#             expected_jac_outer_prod = jnp.matmul(
#                 mu_j, mu_j.T
#             )  # [input_dim x input_dim]
#             assert expected_jac_outer_prod.shape == (input_dim, input_dim)

#             # expected_metric_tensor = expected_jac_outer_prod + cov_weight * output_dim * cov_j
#             expected_metric_tensor = (
#                 expected_jac_outer_prod + cov_weight * cov_j
#             )
#             # expected_metric_tensor = cov_weight * cov_j.T
#             # expected_metric_tensor = cov_j
#             assert expected_metric_tensor.shape == (input_dim, input_dim)
#             return expected_metric_tensor, mu_j, cov_j

#         num_test_inputs = Xnew.shape[0]
#         input_dim = Xnew.shape[1]
#         output_dim = f.shape[1]
#         print("inside gp_metric_tensor")
#         print(output_dim)

#         expected_metric_tensor, mu_j, cov_j = jax.vmap(
#             calc_expected_metric_tensor_single, in_axes=(0)
#         )(Xnew)
#         return expected_metric_tensor, mu_j, cov_j


# class SVGPMetricTensor(GPMetricTensor):
#     def __init__(
# self,
#     gp: GPBase,
#     X: InputData,
#     f: OutputData,
#     covariance_weight: jnp.float64 = 1.0,
#     q_sqrt=None,
#     white: bool = True,
#     jitter: jnp.float64 = 1.0e-4,
# ):
#     super().__init__(
#         kernel=kernel,
#         mean_function=mean_function,
#         covariance_weight=covariance_weight,
#         jitter=jitter,
#     )
#     full_cov = True

# def metric_fn(
#     self,
#     Xnew: InputData,
#     X: InputData,
#     f: OutputData,
# ):
#     return 0
