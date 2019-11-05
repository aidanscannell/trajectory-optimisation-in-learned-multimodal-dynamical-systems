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

import numpy as np
import tensorflow as tf
from gpflow import settings, transforms
from gpflow.decors import params_as_tensors
from gpflow.likelihoods import Bernoulli, Likelihood
from gpflow.params import Parameter, ParamList


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class BernoulliGaussian(Likelihood):
    """
    Implementation of the following likelihood:

    p(Y | F, a) = {\prod_{n=1}^{N}} \mathcal{N} (y_n|f_1, var_1)^{a_n} \mathcal{N}	(y_n|f_2, var_2)^{1 - a_n}

    where
        Y represents the observed outputs Y = {y_n}_{n=1}^N,
        F = {f_1, f_2}, f_k represents the GP for mode k,
        var_k is the varince of mode k
        here k=2
    """
    def __init__(self,
                 variance_low=np.array([[0.005, 0.], [0., 0.005]]),
                 variance_high=np.array([[0.3, 0.], [0., 0.3]]),
                 invlink=inv_probit,
                 var_trainable=True,
                 **kwargs):
        """
        Parameters
        ----------
        variance_low: noise covariance matrix 1
        variance_high:
        invlink: the inverse link function to be used, default is inverse probit
        var_trainable: if False then variances not trainable, default True
        """
        super().__init__(**kwargs)
        self.invlink = invlink
        variance_low = Parameter(variance_low,
                                 transform=transforms.positive,
                                 dtype=settings.float_type)
        variance_high = Parameter(variance_high,
                                  transform=transforms.positive,
                                  dtype=settings.float_type)

        variances = [variance_low, variance_high]
        self.variances = ParamList(variances)
        self.likelihood_bern = Bernoulli()
        self.input_dim = variance_low.shape[1]

        if var_trainable is False:
            self.variance_low.trainable = False
            self.variance_high.trainable = False

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar, idx):
        if self.input_dim == 1:
            return tf.identity(Fmu), Fvar + tf.squeeze(self.variances[idx])
        else:
            return tf.identity(Fmu), Fvar + tf.diag_part(
                tf.squeeze(self.variances[idx]))

    @params_as_tensors
    def predict_mean_and_var_a(self, Hmu, Hvar):
        return self.likelihood_bern.predict_mean_and_var(Hmu, Hvar)

    @params_as_tensors
    def predict_mean_a(self, H):
        return inv_probit(H)
