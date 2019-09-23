import tensorflow as tf
import numpy as np
from gpflow import transforms, settings
from gpflow.params import Parameter
from gpflow.likelihoods import Likelihood, Bernoulli
from gpflow.decors import params_as_tensors


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class BernoulliGaussian(Likelihood):
    def __init__(self,
                 variance_low=0.005,
                 variance_high=0.3,
                 invlink=inv_probit,
                 **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink
        self.variance_low = Parameter(variance_low,
                                      transform=transforms.positive,
                                      dtype=settings.float_type)
        self.variance_high = Parameter(variance_high,
                                       transform=transforms.positive,
                                       dtype=settings.float_type)
        self.likelihood_bern = Bernoulli()


#         self.variance_low.trainable = False
#         self.variance_high.trainable = False

    @params_as_tensors
    def predict_mean_and_var_f_high(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance_high

    @params_as_tensors
    def predict_mean_and_var_f_low(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance_low

    @params_as_tensors
    def predict_mean_and_var_a(self, Hmu, Hvar):
        return self.likelihood_bern.predict_mean_and_var(Hmu, Hvar)

    @params_as_tensors
    def predict_mean_a(self, H):
        return inv_probit(H)
