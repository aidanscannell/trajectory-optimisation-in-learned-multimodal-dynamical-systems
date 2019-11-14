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
import tensorflow as tf

from gen_data import func, func1, func2, gen_data
from model import BMNSVGP
from utils import plot_a, plot_contourf, plot_loss, plot_model, run_adam

float_type = tf.float64

input_dim = 2
output_dim = 2

num_iters = 15000

var_low = np.array([[[0.005, 0.], [0., 0.005]]])
var_high = np.array([[[0.3, 0.], [0., 0.3]]])
vars_list = [var_low, var_high]

# generate data set
if input_dim is 1 and output_dim is 1:
    func_list = [func]
elif input_dim is 2 and output_dim is 1:
    func_list = [func1]
elif input_dim is 2 and output_dim is 2:
    func_list = [func1, func2]
N = 900  # number of training observations

X, Y, a = gen_data(N,
                   input_dim,
                   output_dim,
                   low_lim=-2.,
                   high_lim=2.,
                   func_list=func_list,
                   plot_flag=False)

# standardise input
X_ = (X - X.mean()) / X.std()
Y_ = (Y - Y.mean()) / Y.std()

gpflow.reset_default_graph_and_session()
with gpflow.defer_build():
    m = BMNSVGP(X_, Y_, noise_vars=vars_list, minibatch_size=100)
#     m = BMNSVGP(X, Y, var_low=0.001, var_high=0.7, minibatch_size=100)
m.compile()

# plot_model(m, m1=True, y=True)
# plot_model(m, a=True, h=True)
# plot_model(m, f=True)
# plot_model(m, y=True)

logger = run_adam(m, maxiter=gpflow.test_util.notebook_niter(num_iters))
plot_loss(logger)

# plot_model(m, a=True, h=True)
# plot_model(m, f=True)
# plot_model(m, y=True)
# print('Finished optimising model.')

# plot_model(m, m1=True, m2=True)
# plot_a(m, a)
# plot_model(m, a=True)
# plot_model(m, y=True)

# saver = gpflow.saver.Saver()
# saver.save('../saved_models/bmnsvgp', m)
