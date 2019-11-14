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

# generate data set
# input_dim = 2
# output_dim = 2
# if input_dim is 1 and output_dim is 1:
#     func_list = [func]
# elif input_dim is 2 and output_dim is 1:
#     func_list = [func1]
# elif input_dim is 2 and output_dim is 2:
#     func_list = [func1, func2]
# N = 900  # number of training observations

# X, Y, a = gen_data(N,
#                    input_dim,
#                    output_dim,
#                    low_lim=-2.,
#                    high_lim=2.,
#                    func_list=func_list,
#                    plot_flag=False)


def train_model(X, Y, num_iters=15000, vars_list=None, minibatch_size=100):
    if vars_list is None:
        dim = Y.shape[1]
        vars_list = [
            np.array([0.005 * np.eye(dim)]),
            np.array([0.3 * np.eye(dim)])
        ]

    # standardise input
    X_ = (X - X.mean()) / X.std()
    Y_ = (Y - Y.mean()) / Y.std()

    gpflow.reset_default_graph_and_session()
    with gpflow.defer_build():
        m = BMNSVGP(X_,
                    Y_,
                    noise_vars=vars_list,
                    minibatch_size=minibatch_size)
    m.compile()

    logger = run_adam(m, maxiter=gpflow.test_util.notebook_niter(num_iters))
    return m, logger


# plot_model_2d_uav(m, X, f=False, a=True, h=False, y=False, y_a=None, var=False)
# plot_model_2d_uav(m, X, f=True , a=False, h=False, y=False, y_a=None, var=False)

# saver = gpflow.saver.Saver()
# saver.save('../saved_models/bmnsvgp', m)

if __name__ == "__main__":
    data = np.load('../data/uav_data1.npz')
    X = data['x']
    Y = data['y']

    m, logger = train_model(X, Y)

    plot_loss(logger)
