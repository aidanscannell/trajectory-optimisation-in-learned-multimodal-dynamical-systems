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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import BMNSVGP
# from utils.gen_data import func, func1, func2, gen_data
from utils.utils import (plot_a, plot_and_save_all, plot_contourf, plot_loss,
                         plot_model, plot_model_2d_uav, run_adam)

float_type = tf.float64


def save_model(m, X, a_true, X_missing, filename="model"):
    import datetime
    from pathlib import Path
    date = datetime.datetime.now()
    date_str = str(date.day) + "-" + str(date.month) + "/" + str(
        date.hour) + str(date.minute) + "/"
    Path("./saved_models/" + date_str).mkdir(parents=True, exist_ok=True)

    lengthscales = m.kern_h.lengthscales.value
    variance = m.kern_h.variance.value
    mX = m.X.value
    mY = m.Y.value
    z = m.feature_h.Z.value
    q_mu = m.q_mu_h.value
    q_sqrt = m.q_sqrt_h.value

    try:
        m.mean_function_h.c.value
    except AttributeError:
        var_exists = False
        print('Constant mean function does NOT exist')
    else:
        var_exists = True
        print('Constant mean function does exist')
    if var_exists:
        mean_function = m.mean_function_h.c.value
    else:
        mean_function = 0.

    N = 100
    x1_high = m.X.value[:, 0].max()
    x2_high = m.X.value[:, 1].max()
    x1_low = m.X.value[:, 0].min()
    x2_low = m.X.value[:, 1].min()

    xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([xx.flat, yy.flat])

    h_mu, h_var = m.predict_h(xy)  # mean and var of alpha
    m_h_mu, m_h_var = m.predict_h(mX)  # mean and var of alpha
    # print(a_mu.shape)
    # print(a_var.shape)

    # save numpy file
    np.savez('saved_models/' + date_str + '/params_from_model',
             l=lengthscales,
             var=variance,
             x=X,
             y=mY,
             z=z,
             q_mu=q_mu,
             q_sqrt=q_sqrt,
             a_true=a_true,
             mean_func=mean_function,
             m_h_mu=m_h_mu,
             m_h_var=m_h_var,
             xx=xx,
             yy=yy,
             xy=xy,
             h_mu=h_mu,
             h_var=h_var)

    # save gpflow model
    saver = gpflow.saver.Saver()
    saver.save('./saved_models/' + date_str + '/model', m)


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
    print("Finished training model.")
    print(m.as_pandas_table())
    return m, logger


if __name__ == "__main__":
    import os
    print(os.getcwd())
    x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 1000
    y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 1000
    # data = np.load('../data/npz/model_inputs_combined_old.npz')
    # data = np.load('../data/npz/model_inputs_combined_z.npz')
    # data = np.load('../data/npz/model_data.npz')
    data = np.load('../data/npz/turbulence/model_data.npz')
    data = np.load('../data/npz/turbulence/model_data_fan_fixed_subset.npz')
    # data = np.load('../data/npz/model_inputs_combined.npz')
    X = data['x']
    Y = data['y']
    # Y = Y[:, 0:2]
    Y = Y[:, 0:1]
    Y[:, 0] = Y[:, 0] * 5
    print(X.shape)
    print(Y.shape)
    plt.quiver(X[:, 0],
               X[:, 1],
               Y[:, 0],
               np.zeros([*Y[:, 0].shape]),
               angles='xy',
               scale_units='xy',
               width=0.001,
               scale=1,
               zorder=10)
    try:
        plt.quiver(X[:, 0],
                   X[:, 1],
                   np.zeros([*Y[:, 0].shape]),
                   Y[:, 1],
                   angles='xy',
                   scale_units='xy',
                   width=0.001,
                   scale=1,
                   zorder=10)
    except IndexError:
        print('Only one output dimension so not plotting it.')
    plt.show()

    # remove some data points
    # mask_0 = X[:, 0] < 0
    # mask_1 = X[:, 1] > -0
    # mask = mask_0 | mask_1
    # x1_low = 0.
    # x2_low = -3.
    # x1_high = 2.5
    # x2_high = -1.2
    x1_low = -3.
    x2_low = -3.
    x1_high = 0.
    x2_high = -1.
    mask_0 = X[:, 0] < x1_low
    mask_1 = X[:, 1] < x2_low
    mask_2 = X[:, 0] > x1_high
    mask_3 = X[:, 1] > x2_high
    mask = mask_0 | mask_1 | mask_2 | mask_3
    X_partial = X[mask, :]
    Y_partial = Y[mask, :]
    # X_partial = X
    # Y_partial = Y
    x1 = [x1_low, x1_low, x1_high, x1_high, x1_low]
    x2 = [x2_low, x2_high, x2_high, x2_low, x2_low]
    X_missing = [x1, x2]

    plt.scatter(X_partial[:, 0], X_partial[:, 1], color='k', marker='x')
    # plt.plot(x_alpha, y_alpha, 'r')
    plt.plot(x1, x2, 'g')
    plt.show()

    print('Y_partial.shape')
    print(Y_partial.shape)

    dim = Y_partial.shape[1]
    vars_list = [
        np.array([0.005 * np.eye(dim)]),
        np.array([0.3 * np.eye(dim)])
    ]

    # m, logger = train_model(X, Y, num_iters=15000)
    # m, logger = train_model(X, Y, num_iters=7000)
    # m, logger = train_model(X_partial, Y_partial, num_iters=150)
    m, logger = train_model(X_partial, Y_partial, num_iters=15000)

    plot_loss(logger)

    # plot_and_save_all(m, X_partial, a_true=[x_alpha, y_alpha])
    plot_and_save_all(m, X_partial)

    # plot_model_2d_uav(m,
    #                   X_partial,
    #                   f=False,
    #                   a=True,
    #                   a_true=[x_alpha, y_alpha],
    #                   h=False,
    #                   y=False,
    #                   y_a=None,
    #                   var=False)

    # saver = gpflow.saver.Saver()
    # saver.save('./saved_models/model_feb6_missing_data_2d_ouput', m)
    # print('successfully past save')

    save_model(m, X_partial, [x_alpha, y_alpha], X_missing)
