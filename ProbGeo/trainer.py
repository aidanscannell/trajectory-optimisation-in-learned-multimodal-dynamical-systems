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


def add_arrow(line,
              position=None,
              direction='right',
              size=15,
              color=None,
              alpha=1):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
        position = ydata.mean()
    # find closest index
    epsilon = 0.09
    cond1 = (ydata > position - epsilon)
    cond2 = (ydata < position + epsilon)
    cond = cond1 & cond2
    # start_ind = np.argmin(np.absolute(xdata - position))
    # start_ind = np.argmin(np.absolute(ydata - position))
    start_inds = np.where(cond)[0]
    print(start_inds)
    if direction == 'right':
        # end_ind = start_ind + 1
        end_inds = start_inds + 1
    else:
        # end_ind = start_ind - 1
        end_inds = start_inds - 1

    for start_ind, end_ind in zip(start_inds, end_inds):
        line.axes.annotate('',
                           xytext=(xdata[start_ind], ydata[start_ind]),
                           xy=(xdata[end_ind], ydata[end_ind]),
                           arrowprops=dict(arrowstyle="->", color=color),
                           size=size,
                           alpha=alpha)


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

    num_dynamics_gps = m.num_dynamics_gps
    input_dim = m.input_dim
    num_inducing = m.num_inducing
    output_dim = m.output_dim

    lengthscales_f = np.empty([num_dynamics_gps, output_dim, input_dim])
    variances_f = np.empty([num_dynamics_gps, output_dim])
    z_f = np.empty([num_dynamics_gps, num_inducing, input_dim])
    lik_f = np.empty([num_dynamics_gps, output_dim])
    meanfncs_f = np.empty([num_dynamics_gps, 1])
    # meanfncs_f = np.empty([num_dynamics_gps, output_dim])
    q_mus_f = np.empty([num_dynamics_gps, num_inducing, output_dim])
    q_sqrts_f = np.empty(
        [num_dynamics_gps, output_dim, num_inducing, num_inducing])

    for mode, (variance_lik, q_mu_f, q_sqrt_f, features_f, kerns,
               mean_funcs_f) in enumerate(
                   zip(m.likelihood.variances, m.q_mus, m.q_sqrts, m.features,
                       m.kernels, m.mean_functions)):
        # var = tf.squeeze(variance_lik)
        lik_f[mode, :] = variance_lik.value
        q_sqrts_f[mode, :] = q_sqrt_f.value
        q_mus_f[mode, :] = q_mu_f.value
        meanfncs_f[mode, :] = mean_funcs_f.c.value
        for feat in features_f.feat_list:
            z = feat.Z.value
            z_f[mode, :, :] = z
        for output, kern in enumerate(
                kerns.kernels):  # iterate through output dimensions
            lengthscales_f[mode, output, :] = kern.lengthscales.value
            variances_f[mode, output] = kern.variance.value
    print(lik_f.shape)
    print(q_mus_f.shape)
    print(q_sqrts_f.shape)
    print(z_f.shape)
    print(lengthscales_f.shape)
    print(variances_f.shape)
    print(meanfncs_f.shape)

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
    np.savez('saved_models/' + date_str + '/params_from_model_h',
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

    np.savez('saved_models/' + date_str + '/params_from_model_f',
             l=lengthscales_f,
             kern_var=variances_f,
             lik_var=lik_f,
             x=X,
             y=mY,
             z_f=z_f,
             q_mu_f=q_mus_f,
             q_sqrts_f=q_sqrts_f,
             mean_func_f=meanfncs_f)

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
    # Y[:, 0] = Y[:, 0] * 5
    # print(X.shape)
    # print(Y.shape)

    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # # contf = plt.tricontourf(X[:, 0], X[:, 1], Y.reshape(-1), 15)
    # ax.set_xlabel("$x$")
    # ax.set_ylabel("$y$")
    # # ax.set_title("")
    # contf = plt.tricontourf(X[:, 0], X[:, 1], Y.reshape(-1), 100)
    # cbar = fig.colorbar(contf, shrink=0.5, aspect=5, ax=ax)
    # cbar.set_label("$\Delta x$")
    # plt.savefig("../images/data/quadcopter-dataset-contour.pdf",
    #             transparent=True,
    #             bbox_inches='tight')
    # ax.scatter(X[:, 0], X[:, 1], marker='x', color='k', alpha=0.7, s=0.5)
    # print('here')
    # num = 380
    # line = ax.plot(X[0:num, 0], X[0:num, 1], color='k', alpha=0.5)[0]
    # add_arrow(line, color='k', alpha=0.5)
    # plt.savefig("../images/data/quadcopter-dataset-contour-with-scatter.pdf",
    #             transparent=True,
    #             bbox_inches='tight')

    # plt.quiver(X[:, 0],
    #            X[:, 1],
    #            Y[:, 0],
    #            np.zeros([*Y[:, 0].shape]),
    #            angles='xy',
    #            scale_units='xy',
    #            width=0.001,
    #            scale=1,
    #            zorder=10)
    try:
        plt.quiver(X[:, 0],
                   X[:, 1],
                   Y[:, 0],
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
