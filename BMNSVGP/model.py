import numpy as np
import tensorflow as tf
import gpflow
from likelihood import BernoulliGaussian
from gpflow import Minibatch, settings, features, transforms, kullback_leiblers
from gpflow.models.model import Model
from gpflow import params_as_tensors, autoflow
from gpflow.params import Parameter
from gpflow.mean_functions import Zero
from gpflow.conditionals import Kuu, conditional
from likelihood import inv_probit


class BMNSVGP(Model):
    def __init__(self, X, Y, var_low, var_high, minibatch_size=None):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        """
        Model.__init__(self, name="BMNSVGP")
        if minibatch_size is not None:
            self.X = Minibatch(X, batch_size=minibatch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
        else:
            self.X = Dataholder(X)
            self.Y = Dataholder(Y)

        self.num_data = X.shape[0]
        self.whiten = True
        #         num_inducing = len(self.feature)

        # init mean functions
        self.mean_function_h = Zero(output_dim=1)
        self.mean_function_f_low = Zero(output_dim=Y.shape[1])
        self.mean_function_f_high = Zero(output_dim=Y.shape[1])

        # init kernels
        self.kern_h = gpflow.kernels.RBF(input_dim=X.shape[1])
        self.kern_f_low = gpflow.kernels.RBF(input_dim=X.shape[1])
        self.kern_f_high = gpflow.kernels.RBF(input_dim=X.shape[1])

        # init sparse features
        feat = None
        M = 50
        #         idx = np.random.choice(range(X.shape[0]), size=M, replace=False).reshape(-1, 1)
        #         X_ = tf.layers.flatten(X)
        idx = np.random.choice(range(X.shape[0]), size=M, replace=False)
        Z = X[idx, ...].reshape(-1, 1)
        self.feature_h = features.inducingpoint_wrapper(feat, Z)
        self.feature_h.trainable = False
        self.feature_f_low = features.inducingpoint_wrapper(feat, Z)
        self.feature_f_low.trainable = False
        self.feature_f_high = features.inducingpoint_wrapper(feat, Z)
        self.feature_f_high.trainable = False

        # init likelihood
        self.likelihood = BernoulliGaussian(variance_low=var_low,
                                            variance_high=var_high)

        # init variational parameters
        # TODO: auto select number of inducing points
        q_mu_h = np.zeros((50, 1)) + np.random.randn(50, 1)
        #         q_mu_h = np.zeros((50, 1)) + np.random.randn(50, 1) * 10
        q_sqrt_h = np.array(
            [10 * np.eye(50, dtype=settings.float_type) for _ in range(1)])
        #         q_mu_h = np.zeros((50, 1))
        #         q_mu_h[Z>0] += -1
        #         q_mu_h[Z<0] += 100
        self.q_mu_h, self.q_sqrt_h = self._init_variational_parameters(
            num_inducing=50, num_latent=1, q_mu=q_mu_h, q_sqrt=q_sqrt_h)
        #         self.q_mu_h, self.q_sqrt_h = self._init_variational_parameters(num_inducing=50, num_latent=1)
        self.q_mu_f_low, self.q_sqrt_f_low = self._init_variational_parameters(
            num_inducing=50, num_latent=self.Y.shape[1])
        self.q_mu_f_high, self.q_sqrt_f_high = self._init_variational_parameters(
            num_inducing=50, num_latent=self.Y.shape[1])

    def _init_variational_parameters(self,
                                     num_inducing,
                                     num_latent,
                                     q_mu=None,
                                     q_sqrt=None,
                                     q_diag=None):
        q_mu = np.zeros((num_inducing, num_latent)) if q_mu is None else q_mu
        #         q_mu = q_mu + np.random.randn(q_mu.shape[0], q_mu.shape[1])
        q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if q_diag:
                #                 q_sqrt = q_sqrt + np.random.randn(q_sqrt.shape[0], q_sqrt.shape[1])
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

    def _sample_e_h(self, dist_h, f_low, f_high, num_samples=1000):
        h = tf.reshape(dist_h.sample(num_samples), [-1, num_samples])
        p_a_0 = 1 - inv_probit(h)

        dist_y_low = tf.distributions.Normal(
            loc=f_low, scale=self.likelihood.variance_low)
        dist_y_high = tf.distributions.Normal(
            loc=f_high, scale=self.likelihood.variance_high)
        p_y_low = dist_y_low.prob(self.Y)
        p_y_high = dist_y_high.prob(self.Y)
        return 1. / (num_samples**2) * tf.reduce_sum(
            tf.log(p_y_low * p_a_0 + p_y_high * (1 - p_a_0)), axis=1)

    def _sample_e_f(self, dist_f_low, dist_f_high, h, num_samples=1000):
        p_a_0 = 1 - inv_probit(h)

        f_low = tf.reshape(dist_f_low.sample(num_samples), [-1, num_samples])
        f_high = tf.reshape(dist_f_high.sample(num_samples), [-1, num_samples])
        dist_y_low = tf.distributions.Normal(
            loc=f_low, scale=self.likelihood.variance_low)
        dist_y_high = tf.distributions.Normal(
            loc=f_high, scale=self.likelihood.variance_high)
        p_y_low = dist_y_low.prob(self.Y)
        p_y_high = dist_y_high.prob(self.Y)
        return 1. / (num_samples**2) * tf.reduce_sum(
            tf.log(p_y_low * p_a_0 + p_y_high * (1 - p_a_0)), axis=1)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        # Get prior KL.
        KL_h = self.build_prior_KL(self.feature_h, self.kern_h, self.q_mu_h,
                                   self.q_sqrt_h)
        KL_f_low = self.build_prior_KL(self.feature_f_low, self.kern_f_low,
                                       self.q_mu_f_low, self.q_sqrt_f_low)
        KL_f_high = self.build_prior_KL(self.feature_f_high, self.kern_f_high,
                                        self.q_mu_f_high, self.q_sqrt_f_high)

        # Lets get conditional p(h_n | U_h, x_n) for all N
        h_mean, h_var = self._build_predict_h(self.X,
                                              full_cov=False,
                                              full_output_cov=False)

        # Lets get conditionals p(f_n_1 | U_f_1, x_n) and p(f_n_2 | U_f_2, x_n) for all N
        f_mean_low, f_var_low = self._build_predict_f_low(
            self.X, full_cov=False, full_output_cov=False)
        f_mean_high, f_var_high = self._build_predict_f_high(
            self.X, full_cov=False, full_output_cov=False)

        dist_f_low = tf.distributions.Normal(loc=f_mean_low, scale=f_var_low)
        dist_f_high = tf.distributions.Normal(loc=f_mean_high,
                                              scale=f_var_high)
        dist_h = tf.distributions.Normal(loc=h_mean, scale=h_var)

        f_high = f_mean_high
        f_low = f_mean_low
        var_exp_h = self._sample_e_h(dist_h, f_low, f_high, num_samples=1)

        var_exp_f = self._sample_e_f(dist_f_low,
                                     dist_f_high,
                                     h_mean,
                                     num_samples=1)

        var_exp = var_exp_f + var_exp_h

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(
            tf.shape(self.X)[0], settings.float_type)
        return tf.reduce_sum(var_exp) * scale - KL_f_low - KL_f_high - KL_h

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
    def _build_predict_f_low(self, Xnew, full_cov=False,
                             full_output_cov=False):
        mu, var = conditional(Xnew,
                              self.feature_f_low,
                              self.kern_f_low,
                              self.q_mu_f_low,
                              q_sqrt=self.q_sqrt_f_low,
                              full_cov=full_cov,
                              white=self.whiten,
                              full_output_cov=full_output_cov)
        return mu + self.mean_function_f_low(Xnew), var

    @params_as_tensors
    def _build_predict_f_high(self,
                              Xnew,
                              full_cov=False,
                              full_output_cov=False):
        mu, var = conditional(Xnew,
                              self.feature_f_high,
                              self.kern_f_high,
                              self.q_mu_f_high,
                              q_sqrt=self.q_sqrt_f_high,
                              full_cov=full_cov,
                              white=self.whiten,
                              full_output_cov=full_output_cov)
        return mu + self.mean_function_f_high(Xnew), var

    @autoflow((settings.float_type, [None, None]))
    def predict_a(self, Xnew):
        h_mean, h_var = self._build_predict_h(Xnew)
        return self.likelihood.predict_mean_and_var_a(h_mean, h_var)

    @autoflow((settings.float_type, [None, None]))
    def predict_h(self, Xnew):
        return self._build_predict_h(Xnew)

    @autoflow((settings.float_type, [None, None]))
    def predict_f_low(self, Xnew):
        pred_f_mean, pred_f_var = self._build_predict_f_low(Xnew)
        return self.likelihood.predict_mean_and_var_f_low(
            pred_f_mean, pred_f_var)

    @autoflow((settings.float_type, [None, None]))
    def predict_f_high(self, Xnew):
        pred_f_mean, pred_f_var = self._build_predict_f_high(Xnew)
        return self.likelihood.predict_mean_and_var_f_high(
            pred_f_mean, pred_f_var)

    #     @autoflow((settings.float_type, [None, None]))
    def predict_y(self, Xnew):
        # TODO: how to incorporate unc in assignment???
        pred_f_mean_low, pred_f_var_low = self.predict_f_low(Xnew)
        pred_f_mean_high, pred_f_var_high = self.predict_f_high(Xnew)
        pred_a_mean, pred_a_var = self.predict_a(Xnew)
        f_mean = pred_f_mean_low * (
            1 - pred_a_mean) + pred_f_mean_high * pred_a_mean
        f_var = pred_f_var_low * (1 -
                                  pred_a_mean) + pred_f_var_high * pred_a_mean
        return f_mean, f_var
