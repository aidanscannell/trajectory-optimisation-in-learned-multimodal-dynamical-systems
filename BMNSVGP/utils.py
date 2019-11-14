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
from matplotlib import cm


class Logger(gpflow.actions.Action):
    def __init__(self, model):
        self.model = model
        self.logf = []

    def run(self, ctx):
        if (ctx.iteration % 10) == 0:
            # Extract likelihood tensor from Tensorflow session
            likelihood = -ctx.session.run(self.model.likelihood_tensor)
            # Append likelihood value to list
            self.logf.append(likelihood)


def run_adam(model, maxiter=450):
    # Create an Adam Optimiser action
    adam = gpflow.train.AdamOptimizer().make_optimize_action(model)
    # Create a Logger action
    logger = Logger(model)
    actions = [adam, logger]
    # Create optimisation loop that interleaves Adam with Logger
    loop = gpflow.actions.Loop(actions, stop=maxiter)()
    # Bind current TF session to model
    model.anchor(model.enquire_session())
    return logger


def plot_loss(logger):
    plt.plot(-np.array(logger.logf))
    plt.xlabel('iteration (x10)')
    plt.ylabel('ELBO')
    plt.show()


def plot_data_1d(X, Y, a, func, title=None):
    plt.figure(figsize=(12, 4))
    plt.plot(X, Y, 'x', color='k', alpha=0.4, label="Observations")
    X = X.flatten()
    order = np.argsort(X)
    # plt.plot(X[order], a[order], '-', color='b', alpha=0.4, label="$alpha$")
    aa = np.where(a > 0.5, np.ones(a.shape), np.zeros(a.shape))
    plt.plot(X[order], aa[order], '-', color='k', alpha=0.4, label="$alpha$")
    Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
    Yt = func(Xt)
    plt.plot(Xt, Yt, c='k')  # , label="Underlying function"
    plt.xlabel("$(\mathbf{s}_{t-1}, \mathbf{a}_{t-1})$", fontsize=20)
    plt.ylabel("$\mathbf{s}_t$", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.ylim(-2.1, 2.1)
    plt.legend(loc='lower right', fontsize=15)
    plt.title(title)
    plt.show()


def plot_model(m,
               f=False,
               a=False,
               h=False,
               y=False,
               a_true=None,
               y_true=False,
               y_a=None,
               inputs=False,
               var=False,
               save_name=None):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    # TODO: if colours/labels not given set dynamically
    colours = ['m', 'c']
    labels = ['Mode 1 - Low Noise', 'Mode 2 - High Noise']
    if m.input_dim is 1:
        plot_model_1d(ax, m, f, a, h, y, save_name)
    elif m.input_dim is 2:
        # plot_model_2d(m, f, a, h, y, save_name)
        a_true = False
        y_true = False
        y_a = False
        inputs = False
        var = False
        plot_model_2d(m, f, a, a_true, h, y, y_true, y_a, inputs, var,
                      save_name)

    fig.legend(loc='lower right', fontsize=15)
    plt.xlabel('$(\mathbf{s}_{t-1}, \mathbf{a}_{t-1})$', fontsize=30)
    plt.ylabel('$\mathbf{s}_t$', fontsize=30)
    #     plt.xlim(-1.0, 1.2)
    plt.tick_params(labelsize=20)
    if save_name is not False:
        plt.savefig(save_name, transparent=True, bbox_inches='tight')
    plt.show()


def plot_model_1d(ax, m, f=False, a=False, h=False, y=False, save_name=None):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    pX = np.linspace(-1, 1, 100)[:, None]  # Test locations
    # colours = ['k|', 'b|']

    ax.plot(m.X.value,
            m.Y.value,
            'x',
            color='k',
            label='Observations',
            alpha=0.2)
    # if m1 or m2 or a or h:
    #     for feat, c in zip(m.features[0].feat_list, colours):
    #         plt.plot(feat.Z.value, np.zeros(feat.Z.value.shape), c, mew=2)

    if h is True:
        a_mu, a_var = m.predict_h(pX)  # Predict alpha values at test locations
        ax.plot(pX, a_mu, color='olive', lw=1.5)
        ax.fill_between(pX[:, 0], (a_mu - 2 * a_var**0.5)[:, 0],
                        (a_mu + 2 * a_var**0.5)[:, 0],
                        color='olive',
                        alpha=0.4,
                        lw=1.5,
                        label='Separation manifold GP')

    if a is True:
        a_mu, a_var = m.predict_a(pX)  # Predict alpha values at test locations
        ax.plot(pX, a_mu, color='olive', lw=1.5)
        ax.fill_between(pX[:, 0], (a_mu - 2 * a_var**0.5)[:, 0],
                        (a_mu + 2 * a_var**0.5)[:, 0],
                        color='blue',
                        alpha=0.4,
                        lw=1.5,
                        label='$\\alpha$')

    if f is True:
        pYs, pYvs = m.predict_f(pX)
        #         pY_low, pYv_low = m.predict_f_low(pX)
        for pY, pYv, colour, label in zip(pYs, pYvs, colours, labels):
            line, = ax.plot(pX, pY, color=colour, alpha=0.6, lw=1.5)
            ax.fill_between(pX[:, 0], (pY - 2 * pYv**0.5)[:, 0],
                            (pY + 2 * pYv**0.5)[:, 0],
                            color=colour,
                            alpha=0.2,
                            lw=1.5,
                            label=label)

    if y is True:
        pY, pYv = m.predict_y(pX)
        line, = ax.plot(pX, pY, color='royalblue', alpha=0.6, lw=1.5)
        ax.fill_between(pX[:, 0], (pY - 2 * pYv**0.5)[:, 0],
                        (pY + 2 * pYv**0.5)[:, 0],
                        color='royalblue',
                        alpha=0.2,
                        lw=1.5)


def plot_a(m, a, save_name=False):
    fig = plt.figure(figsize=(12, 4))
    # pX = np.linspace(-1, 1, 100)[:, None]  # Test locations
    pX = np.linspace(m.X.value.min(), m.X.value.max(), 100)[:, None]

    plt.plot(m.X.value, a, '-', color='k', label='True $\\alpha$', alpha=0.9)
    plt.plot(m.feature_f_low.Z.value,
             np.zeros(m.feature_f_low.Z.value.shape),
             'k|',
             mew=2)
    plt.plot(m.feature_f_high.Z.value,
             np.zeros(m.feature_f_high.Z.value.shape),
             'b|',
             mew=2)
    a_mu, a_var = m.predict_a(pX)  # Predict alpha values at test locations
    #     plt.plot(pX, a_mu, color='olive', lw=1.5)
    #     plt.fill_between(pX[:, 0], (a_mu-2*a_var**0.5)[:, 0], (a_mu+2*a_var**0.5)[:, 0], color='blue', alpha=0.4, lw=1.5, label='$\\alpha$')
    plt.plot(pX, -a_mu + 1., color='olive', lw=1.5)
    plt.fill_between(pX[:, 0],
                     -(a_mu - 2 * a_var**0.5)[:, 0] + 1.,
                     -(a_mu + 2 * a_var**0.5)[:, 0] + 1.,
                     color='blue',
                     alpha=0.4,
                     lw=1.5,
                     label='Learnt $\\alpha$')
    fig.legend(loc='lower right', fontsize=15)
    plt.xlabel('$(\mathbf{s}_{t-1}, \mathbf{a}_{t-1})$', fontsize=30)
    plt.ylabel('$\\alpha$', fontsize=30)
    #     plt.xlim(-1.0, 1.2)
    plt.tick_params(labelsize=20)
    if save_name is not False:
        plt.savefig(save_name, transparent=True, bbox_inches='tight')
    plt.show()


def plot_contour(ax, x, y, z, a=None, contour=None, inputs=False):
    # surf = ax.contourf(x, y, z, cmap=cm.cool, linewidth=0, antialiased=False)
    surf = ax.contourf(x,
                       y,
                       z,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
    cont = None
    if contour is not None and a is not None:
        cont = ax.contour(x, y, a, levels=contour)
    if contour is not None and a is None:
        cont = ax.contour(x, y, z, levels=contour)
    if inputs is True:
        # plt.plot(x.flatten(), y.flatten(), 'xk')
        plt.plot(X_[:, 0], X_[:, 1], 'xk')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    return surf, cont


def plot_contourf(x,
                  y,
                  z,
                  a=None,
                  contour=None,
                  title=None,
                  inputs=False,
                  save_name=None):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    surf, cont = plot_contour(ax, x, y, z, a, contour, inputs=inputs)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    if save_name is not None:
        plt.savefig(save_name, transparent=True, bbox_inches='tight')
    plt.show()


def plot_contourf_var(x,
                      y,
                      z,
                      z_var,
                      a=None,
                      contour=None,
                      title=None,
                      inputs=False,
                      save_name=None):
    # fig, axs = plt.subplot(2, sharex=True, sharey=True)
    fig, axs = plt.subplots(1, 2, figsize=(24, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    surf_mu, cont_mu = plot_contour(axs[0], x, y, z, a, contour, inputs)
    cbar = fig.colorbar(surf_mu, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label('mean')
    surf_var, cont_var = plot_contour(axs[1], x, y, z_var, a, contour, inputs)
    cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label('variance')

    plt.suptitle(title)
    if save_name is not None:
        plt.savefig(save_name, transparent=True, bbox_inches='tight')
    plt.show()


def plot_model_2d(m,
                  f=False,
                  a=False,
                  a_true=None,
                  h=False,
                  y=False,
                  y_true=False,
                  y_a=None,
                  inputs=False,
                  var=False,
                  save_name=None):

    N = int(np.sqrt(m.X.value.shape[0]))
    # low = -1; high = 1
    # low = -1.5; high = 1.5
    low = -2
    high = 2
    xx, yy = np.mgrid[low:high:N * 1j, low:high:N * 1j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([xx.flat, yy.flat])

    mx = m.X.value[:, 0].reshape(N, N)
    my = m.X.value[:, 1].reshape(N, N)
    mz1 = m.Y.value[:, 0].reshape(N, N)
    mz2 = m.Y.value[:, 1].reshape(N, N)

    if a is True:
        a_mu, a_var = m.predict_a(xy)  # Predict alpha values at test locations
        # plot_contourf(xx, yy, a_mu.reshape(xx.shape), contour=[0.5], title='predicted alpha')
        # plot_contourf(xx, yy, 1-a_mu.reshape(xx.shape), contour=[0.5], title='predicted alpha inverted')
        plot_contourf_var(xx,
                          yy,
                          a_mu.reshape(xx.shape),
                          a_var.reshape(xx.shape),
                          contour=[0.5],
                          title='predicted alpha',
                          inputs=inputs,
                          save_name="img/learned_alpha" + save_name + ".pdf")
        # plot_contourf_var(xx, yy, 1-a_mu.reshape(xx.shape), a_var.reshape(xx.shape), contour=[0.5], title='predicted alpha inverted', inputs=inputs)

    if a_true is not None:
        plot_contourf(mx,
                      my,
                      a_true.reshape(xx.shape),
                      contour=[0.5],
                      title='Original alpha',
                      inputs=inputs,
                      save_name="img/original_alpha" + save_name + ".pdf")

    if f is True:
        f_mus, f_vars = m.predict_f(
            xy)  # Predict alpha values at test locations
        for i, (f_mu, f_var) in enumerate(zip(f_mus, f_vars)):
            plot_contourf_var(xx,
                              yy,
                              f_mu[:, 0].reshape(xx.shape),
                              f_var[:, 0].reshape(xx.shape),
                              title=('predicted $f_%i$ dim 1' % i),
                              inputs=inputs)
            plot_contourf_var(xx,
                              yy,
                              f_mu[:, 1].reshape(xx.shape),
                              f_var[:, 1].reshape(xx.shape),
                              title=('predicted $f_%i$ dim 2' % i),
                              inputs=inputs)

    if y is True:
        y_mu, y_var = m.predict_y(xy)  # Predict alpha values at test locations
        print(y_mu.shape)
        if y_a is not None:
            # plot_contourf(xx, yy, y_mu.reshape(xx.shape), a=y_a.reshape(xx.shape), contour=[0.5], title='predicted y')
            # plot_contourf_var(xx, yy, y_mu.reshape(xx.shape), y_var.reshape(xx.shape), title='predicted y', inputs=inputs)
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 0].reshape(xx.shape),
                              y_var[:, 0].reshape(xx.shape),
                              title='predicted y dim 1',
                              inputs=inputs)
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 1].reshape(xx.shape),
                              y_var[:, 1].reshape(xx.shape),
                              title='predicted y dim 2',
                              inputs=inputs)
        else:
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 0].reshape(xx.shape),
                              y_var[:, 0].reshape(xx.shape),
                              title='predicted y dim 1',
                              inputs=inputs)
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 1].reshape(xx.shape),
                              y_var[:, 1].reshape(xx.shape),
                              title='predicted y dim 2',
                              inputs=inputs)
            # plot_contourf(xx, yy, y_mu.reshape(xx.shape), y_var.reshape(xx.shape), title='predicted y', inputs=inputs)
            # TODO: how to plot variance of GPs??

    if y_true is True:
        if y_a is not None:
            plot_contourf(mx,
                          my,
                          mz1,
                          a=y_a.reshape(mx.shape),
                          contour=[0.5],
                          title='original y dim 1',
                          inputs=inputs)
            plot_contourf(mx,
                          my,
                          mz2,
                          a=y_a.reshape(mx.shape),
                          contour=[0.5],
                          title='original y dim 2',
                          inputs=inputs)
        else:
            plot_contourf(mx, my, mz1, title='original y dim 1', inputs=inputs)
            plot_contourf(mx, my, mz2, title='original y dim 2', inputs=inputs)

    if var is True:
        lik_var, f_var = m.predict_vars(xy)
        plot_contourf_var(xx,
                          yy,
                          lik_var.reshape(xx.shape),
                          f_var.reshape(xx.shape),
                          title='noise variance vs GP covariance',
                          inputs=inputs)


# plot_model2D(m, a=True, a_true=a, y=True, y_true=True, y_a=a)
# plot_model2D(m, a=True, a_true=a)
# plot_model2D(m, f=True, y_true=True, y_a=a)
# plot_model2D(m, f=True)
