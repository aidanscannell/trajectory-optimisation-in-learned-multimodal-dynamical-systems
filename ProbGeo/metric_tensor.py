# from abc import ABC, abstractmethod

# import matplotlib.pyplot as plt
# import numpy as np

# class MetricTensorBase(ABC):
#     @abstractmethod
#     def calc_metric_tensor_at_input(self, Xnew):
#         raise NotImplementedError

# class GPMetricTensorBase(MetricTensorBase):
#     def __init__(self, X, Y, gp, var_weight=1):
#         super(GPMetricTensorBase, self).__init__()
#         self.X = X
#         self.Y = Y
#         self.gp = gp
#         self.var_weight = var_weight
#         self.output_dim = self.Y.shape[1]
#         self.input_dim = self.X.shape[1]

#     @abstractmethod
#     def cov_fun(self, x1, x2):
#         raise NotImplementedError

#     # def grad_k_x1(self, x1, x2, full_cov=True):
#     #     # print('inside grad_k_x1')
#     #     with tf.GradientTape() as t:
#     #         # print('inside grad_k_x1 watch')
#     #         t.watch(x1)
#     #         Kxx = self.cov_fun(x1, x2, full_cov=full_cov)
#     #     # print('kxx grad')
#     #     # print(Kxx.shape)
#     #     return tf.squeeze(t.batch_jacobian(Kxx, x1))

#     # def grad_k_x1x2(self, x1, x2, full_cov=True):
#     # # x2 = tf.expand_dims(x2, 1)
#     # with tf.GradientTape() as t:
#     #     t.watch(x2)
#     #     dk_dx1 = self.grad_k_x1(x1, x2, full_cov=full_cov)
#     # return tf.squeeze(t.jacobian(dk_dx1, x2))

#     def calc_jacobian(self, Xnew_all, full_cov=False):
#         def calc_jacobian_at_single_x(Xnew):
#             Xnew = tf.reshape(Xnew, [1, *Xnew.shape])
#             Kxx = self.cov_fun(self.X, self.X)
#             dKdx1 = self.grad_k_x1(Xnew, self.X)
#             print('dKdx1')
#             print(dKdx1.shape)
#             d2Kdx1x2 = self.grad_k_x1x2(Xnew, Xnew)
#             # d2Kdx1x2 += 7

#             lengthscale = self.gp.kernel.lengthscales
#             XnewT = tf.transpose(Xnew, [1, 0])
#             d2Kdx1x2 = self.cov_fun(XnewT, XnewT) / lengthscale[0]**2
#             # d2Kdx1x2 += 1
#             # TODO implement d2k correctly
#             # print("d2Kdx1x2")
#             # print(d2Kdx1x2.shape)
#             # tf.print('d2K')
#             # tf.print(d2Kdx1x2)

#             print('self.X')
#             print(self.X)
#             print(Kxx.shape)
#             Kxx += eye(tf.shape(self.X)[0],
#                        value=default_jitter(),
#                        dtype=self.X.dtype)
#             full_cov = True
#             mu_j, cov_j = base_conditional(dKdx1,
#                                            Kxx,
#                                            d2Kdx1x2,
#                                            self.Y,
#                                            q_sqrt=None,
#                                            full_cov=full_cov)
#             return mu_j + self.gp.mean_function(Xnew), cov_j

#         mu_j, cov_j = tf.map_fn(calc_jacobian_at_single_x,
#                                 Xnew_all,
#                                 dtype=(tf.float64, tf.float64))
#         cov_j = tf.squeeze(cov_j)
#         print('cov_j')
#         print(cov_j.shape)
#         cov_j = tf.linalg.diag_part(cov_j)
#         # print(cov_j.shape)
#         return mu_j, cov_j

#     def calc_metric_tensor_at_input(self, Xnew):
#         mu_j, cov_j = self.calc_jacobian(Xnew)
#         expected_jacobian_outer_prod = tf.matmul(
#             mu_j, mu_j, transpose_b=True)  # [num_test, input_dim x input_dim]
#         print('cov_j 2')
#         print(cov_j.shape)
#         cov_j = tf.linalg.diag(cov_j)
#         print(cov_j.shape)

#         self.expected_metric_tensor = expected_jacobian_outer_prod + self.var_weight * self.output_dim * cov_j  # [input_dim x input_dim]
#         self.expected_metric_tensor = self.expected_metric_tensor + eye(
#             tf.shape(self.expected_metric_tensor)[-1],
#             value=default_jitter(),
#             dtype=self.X.dtype)
#         return self.expected_metric_tensor, mu_j, cov_j

#     def calc_vec_metric_tensor_at_input(self, Xnew):
#         expected_metric_tensor, _, _ = self.calc_metric_tensor_at_input(Xnew)
#         return tf.reshape(expected_metric_tensor,
#                           [tf.shape(expected_metric_tensor)[0], -1])

# class GPMetricTensor(GPMetricTensorBase):
#     def __init__(self, gp, X, Y, var_weight=1):
#         super(GPMetricTensor, self).__init__(X, Y, gp, var_weight)

#     def cov_fun(self, x1, x2, full_cov=True):
#         # TODO does full_cov need to be here?
#         return self.gp.kernel.K(x1, x2)

# def load_npz_and_init_gp(
#         filename='../models/saved_model/params_artificial_gp.npz'):
#     params = np.load(filename)
#     lengthscales = params['l']
#     kern_var = params['var']
#     mean_func = params['mean_func']
#     X = params['x']
#     Y = params['a']
#     X = tf.convert_to_tensor(X)
#     Y = tf.convert_to_tensor(Y)

#     lengthscales = tf.convert_to_tensor(lengthscales, dtype=default_float())
#     variance = tf.convert_to_tensor(kern_var, dtype=default_float())
#     kern = gpf.kernels.RBF(lengthscales=lengthscales, variance=variance)
#     mean_func = gpf.mean_functions.Constant()
#     m = gpf.models.GPR(data=(X, Y), kernel=kern, mean_function=mean_func)

#     m.likelihood.variance.assign(0.00001)

#     return m, X, Y

# if __name__ == '__main__':
#     from src.visualisation.gp import plot_jacobian_mean, plot_jacobian_var, plot_mean_and_var
#     from src.visualisation.metric import plot_metric_trace
#     var_weight = 3.8
#     var_weight = 38
#     # var_weight = 100
#     filename = '../models/saved_model/params_artificial_gp.npz'

#     gp, X, Y = load_npz_and_init_gp(filename)
#     print_summary(gp)

#     metric_tensor = GPMetricTensor(gp, X, Y, var_weight=var_weight)

#     # create test inputs
#     Xnew1 = np.linspace(-3.2, 3.2, 10)
#     Xnew2 = np.linspace(-3.2, 3.2, 10)
#     xx, yy = np.meshgrid(Xnew1, Xnew2)
#     Xnew = np.stack([xx.flatten(), yy.flatten()]).T
#     # Xnew = np.array([1.0, 3.0]).reshape(1, 2)
#     Xnew = tf.convert_to_tensor(Xnew)

#     # predict GP predictions at Xnew
#     mu, var = gp.predict_f(Xnew)
#     # fig, axs = plot_mean_and_var(xx, yy, mu.numpy(), var.numpy())
#     # fig.suptitle('SVGP Predictive Posterior')

#     expected_metric_tensor, mu_j, cov_j = metric_tensor.calc_metric_tensor_at_input(
#         Xnew)
#     print('mu_j, cov_j')
#     print(mu_j.shape)
#     print(cov_j.shape)
#     print('G')
#     print(expected_metric_tensor.shape)

#     fig, axs = plot_jacobian_mean(xx, yy, Xnew.numpy(), mu_j.numpy(),
#                                   mu.numpy(), var.numpy())
#     fig.suptitle("$\mathbb{E}[\mathbf{J}] = \mathbf{\mu}_J$")
#     fig, axs = plot_jacobian_var(xx, yy, Xnew, cov_j.numpy())
#     fig.suptitle("$\mathbb{V}[\mathbf{J}] = diag(\Sigma_J)$")
#     fig, axs = plot_jacobian_var(xx, yy, Xnew, expected_metric_tensor.numpy())
#     fig.suptitle("$\mathbb{E}[\mathbf{G}]$")
#     fig, axs = plot_metric_trace(xx, yy, expected_metric_tensor.numpy())
#     plt.show()
