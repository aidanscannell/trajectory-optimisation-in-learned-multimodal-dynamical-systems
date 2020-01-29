# Copyright 2019 Aidan Scannell

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gpflow
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import (autoflow, features, kullback_leiblers, params_as_tensors,
                    settings, transforms)
from gpflow.conditionals import Kuu, conditional
# from gpflow.decors import autoflow, params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.models.model import Model
from gpflow.multioutput import features as mf
from gpflow.multioutput import kernels as mk
from gpflow.params import DataHolder, Minibatch, Parameter, ParamList

from derivative_kernel import SquaredExponentialDerivative
# from derivative_kernel import SquaredExponential
from likelihood import BernoulliGaussian, inv_probit

float_type = gpflow.settings.float_type


class BMNSVGP(Model):
    """ Bimodal Noise Sparse Variational Gaussian Process Class. """
    def __init__(self,
                 X,
                 Y,
                 noise_vars,
                 minibatch_size=None,
                 var_trainable=True):
        """
        Parameters
        ----------
            X: Inputs data matrix, size [num_data x input_dim]
            Y: Outputs data matrix, size [num_data x output_dim]
            noise_vars: List of covariance matrices, one for each mode of the
                        likelihood. Each mode is a Gaussian
                        mode of size [1, input_dim x input_dim]
            minibatch: Size of minibatch for stochastic variational inference.
        """
        Model.__init__(self, name="BMNSVGP")
        assert X.shape[0] == Y.shape[0]
        assert len(noise_vars) == len(Y.shape) == len(X.shape)
        # assert noise_vars[0].shape == noise_vars[1].shape == (1, 2, 2)
        if minibatch_size is not None:
            self.X = Minibatch(X, batch_size=minibatch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

        print("input shape: " + str(X.shape))
        print("output shape: " + str(Y.shape))
        self.num_data = X.shape[0]
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.whiten = True
        # self.whiten = False

        # init separation GP
        self.mean_function_h = Zero(output_dim=1)
        # self.kern_h = gpflow.kernels.RBF(input_dim=self.input_dim, ARD=True)
        self.kern_h = SquaredExponentialDerivative(self.input_dim, ARD=True)
        feat = None
        M = 50
        idx = np.random.choice(range(self.num_data), size=M, replace=False)
        Z = X[idx, ...].reshape(-1, self.input_dim)
        self.feature_h = features.inducingpoint_wrapper(feat, Z)
        self.feature_h.trainable = False
        # init variational parameters
        q_mu_h = np.zeros((M, 1)) + rnd.randn(M, 1)
        q_sqrt_h = np.array(
            [10 * np.eye(M, dtype=settings.float_type) for _ in range(1)])
        self.q_mu_h, self.q_sqrt_h = self._init_variational_parameters(
            num_inducing=M, num_latent=1, q_mu=q_mu_h, q_sqrt=q_sqrt_h)
        #         self.q_mu_h, self.q_sqrt_h = self._init_variational_parameters(num_inducing=50, num_latent=1)

        # init 2 dynamics GPs for each output dimension
        kernels, mean_functions, feat_list, q_mus, q_sqrts = [], [], [], [], []

        # init dynamics GPs for each mode
        # TODO: change 2 to the number of dynamics GPs
        for i in range(2):
            # init mean functions
            mean_functions.append(Zero(output_dim=self.output_dim))

            # Create list of kernels for each output
            kern_list = [
                # SquaredExponentialDerivative(self.input_dim, ARD=True)
                # SquaredExponential(self.input_dim, ARD=True)
                gpflow.kernels.RBF(self.input_dim, ARD=True)
                for _ in range(self.output_dim)
            ]
            # Create multioutput kernel from kernel list
            kern = mk.SeparateIndependentMok(kern_list)
            kernels.append(kern)

            # initialisation of inducing input locations, one set of locations per output
            Zs = [
                X[np.random.permutation(len(X))[:M], ...].copy()
                for _ in range(self.output_dim)
            ]
            # initialise as list inducing features
            feature_list = [gpflow.features.InducingPoints(Z) for Z in Zs]
            # create multioutput features from feature_list
            feature = mf.SeparateIndependentMof(feature_list)
            feature.trainable = False
            feat_list.append(feature)

            # init variational inducing points
            q_mu, q_sqrt = self._init_variational_parameters(
                num_inducing=M, num_latent=self.output_dim)
            # q_mu = np.zeros((M, 1)) + rnd.randn(M, 1)
            # q_sqrt = np.array([10*np.eye(M, dtype=settings.float_type) for _ in range(1)])
            q_mus.append(q_mu)
            q_sqrts.append(q_sqrt)

        self.kernels = ParamList(kernels)
        self.features = ParamList(feat_list)
        self.mean_functions = ParamList(mean_functions)
        self.q_mus = ParamList(q_mus)
        self.q_sqrts = ParamList(q_sqrts)

        # init likelihood
        self.likelihood = BernoulliGaussian(variance_low=noise_vars[0],
                                            variance_high=noise_vars[1],
                                            var_trainable=var_trainable)

    def _init_variational_parameters(self,
                                     num_inducing,
                                     num_latent,
                                     q_mu=None,
                                     q_sqrt=None,
                                     q_diag=None):
        q_mu = np.zeros((num_inducing, num_latent)) if q_mu is None else q_mu
        q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if q_diag:
                q_sqrt = Parameter(np.ones((num_inducing, num_latent),
                                           dtype=settings.float_type),
                                   transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([
                    np.eye(num_inducing, dtype=settings.float_type)
                    for _ in range(num_latent)
                ])
                q_sqrt = Parameter(q_sqrt,
                                   transform=transforms.LowerTriangular(
                                       num_inducing, num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                num_latent = q_sqrt.shape[1]
                q_sqrt = Parameter(q_sqrt,
                                   transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                q_sqrt = Parameter(q_sqrt,
                                   transform=transforms.LowerTriangular(
                                       num_inducing,
                                       num_latent))  # L/P x M x M
        return q_mu, q_sqrt

    @params_as_tensors
    def build_prior_KL(self, feature, kern, q_mu, q_sqrt):
        if self.whiten:
            K = None
        else:
            K = Kuu(feature, kern,
                    jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(q_mu, q_sqrt, K)

    @params_as_tensors
    def _sample_e_h(self, dist_h, f_means, num_samples=1000):
        """
        Calculates variational expectation under separation GP using Gibbs sampling.
        \E_{q(h_n)} [log( p(y_n | f_n, \alpha_n) p(\alpha_n | h_n) )], where,
        q(h_n) = \mathcal{N} (h_n^{(1)}|K_{nm}^{(1)}K_{mn}^{(1)}^{-1}U^{(1)}, \tilde{K}^{(1)})
            dist_h: conditional gaussian dist p(h_n | U_h, x_n) ~ N(h_n | m, s)
            f_measns: list (of length k) of [num_data x output_dim] arrays of f values (mean of conditional p(f_n^k | U_f^k, x_n))
            num_samples: number of samples to draw from dist_h
        """
        h = dist_h.sample(num_samples)  # [num_samples x num_data x 1]
        p_a_0 = 1 - inv_probit(h)
        p_a_0 = tf.reshape(p_a_0,
                           [num_samples, -1])  # [num_samples x num_data]
        p_y = []
        for f_mean, variance in zip(f_means, self.likelihood.variances):
            dist_y = tfp.distributions.MultivariateNormalFullCovariance(
                loc=f_mean, covariance_matrix=variance)
            py = dist_y.prob(self.Y)  # [num_data, ]
            p_y.append(py)
        sum = tf.log(p_y[0] * p_a_0 + p_y[1] *
                     (1 - p_a_0))  # [num_samples x num_data]
        return 1. / (num_samples**2) * tf.reduce_sum(
            sum, axis=[0, 1])  # [num_data, ]

    @params_as_tensors
    def _sample_e_f(self, dist_fs, h, num_samples=1000):
        """
        Calculates variational expectation under dynamics GPs using Gibbs sampling.
        \E_{q(f_n)} [log (p(y_n | f_n, \alpha_n) p(\alpha_n | h_n) )], where,
        q(f_n) = \mathcal{N} (f_n^{(1)}|K_{nm}^{(1)}K_{mn}^{(1)}^{-1}U^{(1)}, \tilde{K}^{(1)}) \mathcal{N} (f_n^{(2)}|K_{nm}^{(2)}K_{mn}^{(2)}^{-1}U^{(2)}, \tilde{K}^{(2)})
            dist_fs: list (of length k) of conditional gaussian dists p(f_n^k | U_f^k, x_n) ~ N(f_n^k|m^k, s^k), where k is number of modes
            h: array of h values (mean of conditional p(h_n | U_h, x_n)) [num_data x 1]
            num_samples: number of samples to draw from each distribution
        """
        p_a_0 = 1 - inv_probit(h)
        p_a_0 = tf.reshape(p_a_0, [-1])  # [num_data, ]
        p_y = []
        for i, (dist_f,
                variance) in enumerate(zip(dist_fs,
                                           self.likelihood.variances)):
            f = dist_f.sample(
                num_samples)  # [num_samples x num_data x output_dim]
            dist_y = tfp.distributions.MultivariateNormalFullCovariance(
                loc=f, covariance_matrix=variance)
            py = dist_y.prob(self.Y)  # [num_samples x num_data]
            p_y.append(
                tf.expand_dims(py, i)
            )  # [1 x num_samples x num_data] or [num_samples x 1 x num_data]
        sum = tf.log(p_y[0] * p_a_0 + p_y[1] *
                     (1 - p_a_0))  # [num_samples x num_samples x num_data]
        return 1. / (num_samples**2) * tf.reduce_sum(
            sum, axis=[0, 1])  # [num_data, ]

    @params_as_tensors
    def _build_likelihood(self):
        """
        This functions calculates the variational bound on the model likelihood,

            L = \sum_{n=1}^N \E_{q(H, F, U)} [log\ p(y_n | f_n, \alpha_n) p(\alpha_n | h_n)  ] \\
                - KL( q(U_h) || p(U_h | Z_h)) \\
                - \sum^K_{k=1} KL( q(U_f^{(k)}) || p(U_f^{(k)} | Z_f^{(k)}))

        with the variational posterior,

            q(H, F, U) &= \prod_{n=1}^N p(h_n | U_h, x_n) q(U_h) p(F_n | U_f, x_n) q(U_f) \\
        """
        # Get prior KL.
        KL_h = self.build_prior_KL(self.feature_h, self.kern_h, self.q_mu_h,
                                   self.q_sqrt_h)

        # Lets get conditional p(h_n | U_h, x_n) for all N
        h_mean, h_var = self._build_predict_h(self.X,
                                              full_cov=False,
                                              full_output_cov=False)
        dist_h = tfp.distributions.Normal(loc=h_mean, scale=h_var)

        KL_f = 0
        f_means, f_vars, dist_fs = [], [], []
        for feature, kern, mean_function, q_mu, q_sqrt in zip(
                self.features, self.kernels, self.mean_functions, self.q_mus,
                self.q_sqrts):
            # Get prior KL.
            KL_f += self.build_prior_KL(feature, kern, q_mu, q_sqrt)

            # Lets get conditionals p(f_n_1 | U_f_1, x_n) and p(f_n_2 | U_f_2, x_n) for all N
            f_mean, f_var = self._build_predict_f(self.X,
                                                  feature,
                                                  kern,
                                                  mean_function,
                                                  q_mu,
                                                  q_sqrt,
                                                  full_cov=False,
                                                  full_output_cov=False)
            f_means.append(f_mean)
            f_vars.append(f_var)

            dist_fs.append(tfp.distributions.Normal(loc=f_mean, scale=f_var))

        # Lets calculate the variatonal expectations
        var_exp_h = self._sample_e_h(dist_h, f_means, num_samples=10)
        var_exp_f = self._sample_e_f(dist_fs, h_mean, num_samples=10)
        var_exp = var_exp_f + var_exp_h

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(
            tf.shape(self.X)[0], settings.float_type)
        return tf.reduce_sum(var_exp) * scale - KL_f - KL_h

    @params_as_tensors
    def _build_predict_h(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(Xnew,
                              self.feature_h,
                              self.kern_h,
                              self.q_mu_h,
                              q_sqrt=self.q_sqrt_h,
                              full_cov=full_cov,
                              white=self.whiten,
                              full_output_cov=full_output_cov)
        return mu + self.mean_function_h(Xnew), var

    @params_as_tensors
    def _build_predict_f(self,
                         Xnew,
                         feature,
                         kern,
                         mean_function,
                         q_mu,
                         q_sqrt,
                         full_cov=False,
                         full_output_cov=False):
        mu, var = conditional(Xnew,
                              feature,
                              kern,
                              q_mu,
                              q_sqrt=q_sqrt,
                              full_cov=full_cov,
                              white=self.whiten,
                              full_output_cov=full_output_cov)
        return mu + mean_function(Xnew), var

    @autoflow((settings.float_type, [None, None]))
    def predict_a(self, Xnew):
        h_mean, h_var = self._build_predict_h(Xnew)
        return self.likelihood.predict_mean_and_var_a(h_mean, h_var)

    @autoflow((settings.float_type, [None, None]))
    def predict_h(self, Xnew):
        return self._build_predict_h(Xnew)

    @autoflow((settings.float_type, [None, None]))
    @params_as_tensors
    def predict_f(self, Xnew):
        means, variances = [], []
        for feature, kern, mean_function, q_mu, q_sqrt in zip(
                self.features, self.kernels, self.mean_functions, self.q_mus,
                self.q_sqrts):
            mean, variance = self._build_predict_f(Xnew, feature, kern,
                                                   mean_function, q_mu, q_sqrt)
            means.append(mean)
            variances.append(variance)
        return means, variances

    # @autoflow((settings.float_type, [None, None]))
    @params_as_tensors
    def predict_vars(self, Xnew):
        # TODO: how to propagate unc from a to y
        a_mean, a_var = self.predict_a(Xnew)
        lik_vars = self.likelihood.variances
        f_vars = []
        for idx, (feature, kern, mean_function, q_mu, q_sqrt) in enumerate(
                zip(self.features, self.kernels, self.mean_functions,
                    self.q_mus, self.q_sqrts)):
            _, f_var = self._build_predict_f(Xnew, feature, kern,
                                             mean_function, q_mu, q_sqrt)
            f_vars.append(f_var)
        f_var = f_vars[0] * (1 - a_mean) + f_vars[1] * a_mean
        lik_var = lik_vars[0] * (1 - a_mean) + lik_vars[1] * a_mean
        return f_var, lik_var

    @autoflow((settings.float_type, [None, None]))
    @params_as_tensors
    def predict_vars(self, Xnew):
        f_means, f_vars, y_means, y_vars = [], [], [], []
        for feature, kern, mean_function, q_mu, q_sqrt in zip(
                self.features, self.kernels, self.mean_functions, self.q_mus,
                self.q_sqrts):
            f_mean, f_var = self._build_predict_f(Xnew, feature, kern,
                                                  mean_function, q_mu, q_sqrt)
            f_means.append(f_mean)
            f_vars.append(f_var)
        h_mean, h_var = self._build_predict_h(Xnew)
        a_mean, a_var = self.likelihood.predict_mean_and_var_a(h_mean, h_var)
        noise_vars = self.likelihood.variances
        noise_var = noise_vars[0] * (1 - a_mean) + noise_vars[1] * a_mean
        f_var = f_vars[0] * (1 - a_mean) + f_vars[1] * a_mean
        return noise_var, f_var

    @autoflow((settings.float_type, [None, None]))
    @params_as_tensors
    def predict_y(self, Xnew):
        y_means, y_vars = [], []
        for idx, (feature, kern, mean_function, q_mu, q_sqrt) in enumerate(
                zip(self.features, self.kernels, self.mean_functions,
                    self.q_mus, self.q_sqrts)):
            f_mean, f_var = self._build_predict_f(Xnew, feature, kern,
                                                  mean_function, q_mu, q_sqrt)
            y_mean, y_var = self.likelihood.predict_mean_and_var(
                f_mean, f_var, idx)
            y_means.append(y_mean)
            y_vars.append(y_var)
        h_mean, h_var = self._build_predict_h(Xnew)
        a_mean, a_var = self.likelihood.predict_mean_and_var_a(h_mean, h_var)
        y_mean = y_means[0] * (1 - a_mean) + y_means[1] * a_mean
        y_var = y_vars[0] * (1 - a_mean) + y_vars[1] * a_mean
        return y_mean, y_var
