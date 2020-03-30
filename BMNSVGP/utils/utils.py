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
    plt.show(block=True)


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
               X,
               f=False,
               a=False,
               h=False,
               y=False,
               a_true=None,
               y_true=False,
               y_a=None,
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
        y_true = False
        y_a = False
        var = False
        plot_model_2d(m,
                      X,
                      f=f,
                      a=a,
                      a_true=a_true,
                      h=h,
                      y=y,
                      y_true=y_true,
                      y_a=y_a,
                      var=var,
                      save_name=save_name)

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


def plot_contour(ax, x, y, z, a=None, contour=None):
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
    # if inputs is True:
    #     # plt.plot(x.flatten(), y.flatten(), 'xk')
    #     plt.plot(X_[:, 0], X_[:, 1], 'xk')
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    return surf, cont


def plot_contourf(x, y, z, a=None, contour=None, title=None, save_name=None):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    surf, cont = plot_contour(ax, x, y, z, a, contour)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    if save_name is not None:
        plt.savefig(save_name, transparent=True, bbox_inches='tight')
    # plt.show(block=True)


def plot_contourf_var(x,
                      y,
                      z,
                      z_var,
                      a=None,
                      a_true=None,
                      contour=None,
                      title=None,
                      save_name=None):
    # fig, axs = plt.subplot(2, sharex=True, sharey=True)
    fig, axs = plt.subplots(1, 2, figsize=(24, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    surf_mu, cont_mu = plot_contour(axs[0], x, y, z, a, contour)
    cbar = fig.colorbar(surf_mu, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label('mean')
    surf_var, cont_var = plot_contour(axs[1], x, y, z_var, a, contour)
    cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label('variance')

    if a_true is not None:
        axs[0].plot(a_true[0], a_true[1], 'k')
        axs[1].plot(a_true[0], a_true[1], 'k')

    plt.suptitle(title)
    if save_name is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        surf_mu, cont_mu = plot_contour(ax, x, y, z, a, contour)
        cbar = fig.colorbar(surf_mu, shrink=0.5, aspect=5, ax=ax)
        if a_true is not None:
            # m, logger = train_model(X_partial, Y_partial, num_iters=15000)
            ax.plot(a_true[0], a_true[1], 'k')
        plt.savefig('../images/mean_' + save_name,
                    transparent=True,
                    bbox_inches='tight')
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        surf_var, cont_var = plot_contour(ax, x, y, z_var, a, contour)
        cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=ax)
        if a_true is not None:
            ax.plot(a_true[0], a_true[1], 'k')
        plt.savefig('../images/var_' + save_name,
                    transparent=True,
                    bbox_inches='tight')
        # extent_mean = axs[0].get_window_extent().transformed(
        #     fig.dpi_scale_trans.inverted())
        # extent_var = axs[1].get_window_extent().transformed(
        #     fig.dpi_scale_trans.inverted())

        # fig.savefig('mean_' + save_name, bbox_inches=extent_mean)
        # fig.savefig('var_' + save_name, bbox_inches=extent_var)

        # pad the saved area by 10% in the x-direction and 20% in the y-direction
        # fig.savefig('mean2_' + save_name,
        #             bbox_inches=extent_mean.expanded(1.38, 1.15))
        # fig.savefig('var2_' + save_name,
        #             bbox_inches=extent_var.expanded(1.38, 1.15))

        # plt.savefig(save_name, transparent=True, bbox_inches='tight')
    plt.show()


def plot_model_2d(m,
                  X,
                  f=False,
                  a=False,
                  a_true=None,
                  h=False,
                  y=False,
                  y_true=False,
                  y_a=None,
                  var=False,
                  save_name=None):

    # N = int(np.sqrt(m.X.value.shape[0]))
    # low = -1; high = 1
    # low = -1.5; high = 1.5
    # low = -2
    # high = 2
    # xx, yy = np.mgrid[low:high:N * 1j, low:high:N * 1j]
    # # Need an (N, 2) array of (x, y) pairs.
    # xy = np.column_stack([xx.flat, yy.flat])
    N = 100

    x1_high = m.X.value[:, 0].max()
    x2_high = m.X.value[:, 1].max()
    x1_low = m.X.value[:, 0].min()
    x2_low = m.X.value[:, 1].min()

    xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([xx.flat, yy.flat])

    # rescale inputs
    xx = xx * X.std() + X.mean()
    yy = yy * X.std() + X.mean()

    if a_true is not None or y_true is True:
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
                          save_name="img/learned_alpha" + save_name + ".pdf")
        # plot_contourf_var(xx, yy, 1-a_mu.reshape(xx.shape), a_var.reshape(xx.shape), contour=[0.5], title='predicted alpha inverted')

    if a_true is not None:
        plot_contourf(mx,
                      my,
                      a_true.reshape(xx.shape),
                      contour=[0.5],
                      title='Original alpha',
                      save_name="img/original_alpha" + save_name + ".pdf")

    if f is True:
        f_mus, f_vars = m.predict_f(
            xy)  # Predict alpha values at test locations
        for i, (f_mu, f_var) in enumerate(zip(f_mus, f_vars)):
            plot_contourf_var(xx,
                              yy,
                              f_mu[:, 0].reshape(xx.shape),
                              f_var[:, 0].reshape(xx.shape),
                              title=('predicted $f_%i$ dim 1' % i))
            plot_contourf_var(xx,
                              yy,
                              f_mu[:, 1].reshape(xx.shape),
                              f_var[:, 1].reshape(xx.shape),
                              title=('predicted $f_%i$ dim 2' % i))

    if y is True:
        y_mu, y_var = m.predict_y(xy)  # Predict alpha values at test locations
        if y_a is not None:
            # plot_contourf(xx, yy, y_mu.reshape(xx.shape), a=y_a.reshape(xx.shape), contour=[0.5], title='predicted y')
            # plot_contourf_var(xx, yy, y_mu.reshape(xx.shape), y_var.reshape(xx.shape), title='predicted y')
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 0].reshape(xx.shape),
                              y_var[:, 0].reshape(xx.shape),
                              title='predicted y dim 1')
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 1].reshape(xx.shape),
                              y_var[:, 1].reshape(xx.shape),
                              title='predicted y dim 2')
        else:
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 0].reshape(xx.shape),
                              y_var[:, 0].reshape(xx.shape),
                              title='predicted y dim 1')
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 1].reshape(xx.shape),
                              y_var[:, 1].reshape(xx.shape),
                              title='predicted y dim 2')
            # plot_contourf(xx, yy, y_mu.reshape(xx.shape), y_var.reshape(xx.shape), title='predicted y')
            # TODO: how to plot variance of GPs??

    if y_true is True:
        if y_a is not None:
            plot_contourf(mx,
                          my,
                          mz1,
                          a=y_a.reshape(mx.shape),
                          contour=[0.5],
                          title='original y dim 1')
            plot_contourf(mx,
                          my,
                          mz2,
                          a=y_a.reshape(mx.shape),
                          contour=[0.5],
                          title='original y dim 2')
        else:
            plot_contourf(mx, my, mz1, title='original y dim 1')
            plot_contourf(mx, my, mz2, title='original y dim 2')

    if var is True:
        lik_var, f_var = m.predict_vars(xy)
        plot_contourf_var(xx,
                          yy,
                          lik_var.reshape(xx.shape),
                          f_var.reshape(xx.shape),
                          title='noise variance vs GP covariance')


# plot_model2D(m, a=True, a_true=a, y=True, y_true=True, y_a=a)
# plot_model2D(m, a=True, a_true=a)
# plot_model2D(m, f=True, y_true=True, y_a=a)
# plot_model2D(m, f=True)


def plot_model_2d_uav(m,
                      X,
                      f=False,
                      a=False,
                      a_true=None,
                      h=False,
                      y=False,
                      y_a=None,
                      var=False):
    N = 100
    x1_high = m.X.value[:, 0].max()
    x2_high = m.X.value[:, 1].max()
    x1_low = m.X.value[:, 0].min()
    x2_low = m.X.value[:, 1].min()

    xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([xx.flat, yy.flat])

    # rescale inputs
    xx = xx * X.std() + X.mean()
    yy = yy * X.std() + X.mean()
    Y = m.Y.value

    if a is True:
        print(xy.shape)
        a_mu, a_var = m.predict_a(xy)  # Predict alpha values at test locations
        # plot_contourf(xx, yy, a_mu.reshape(xx.shape), contour=[0.5], title='predicted alpha')
        # plot_contourf(xx, yy, 1-a_mu.reshape(xx.shape), contour=[0.5], title='predicted alpha inverted')
        a_mu = a_mu * Y.std() + Y.mean()
        a_var = a_var * Y.std() + Y.mean()
        plot_contourf_var(xx,
                          yy,
                          a_mu.reshape(xx.shape),
                          a_var.reshape(xx.shape),
                          a_true=a_true,
                          contour=[0.5],
                          save_name='alpha_quadcopter.pdf',
                          title='predicted alpha')
        # plot_contourf_var(xx, yy, 1-a_mu.reshape(xx.shape), a_var.reshape(xx.shape), contour=[0.5], title='predicted alpha inverted')

    if f is True:
        f_mus, f_vars = m.predict_f(
            xy)  # Predict alpha values at test locations
        for i, (f_mu, f_var) in enumerate(zip(f_mus, f_vars)):
            i = i + 1
            plot_contourf_var(xx,
                              yy,
                              f_mu[:, 0].reshape(xx.shape),
                              f_var[:, 0].reshape(xx.shape),
                              save_name='f' + str(i) + '_dim_1_quadcopter.pdf',
                              title=('predicted $f_%i$ dim 1' % i))
            plot_contourf_var(xx,
                              yy,
                              f_mu[:, 1].reshape(xx.shape),
                              f_var[:, 1].reshape(xx.shape),
                              save_name='f' + str(i) + '_dim_2_quadcopter.pdf',
                              title=('predicted $f_%i$ dim 2' % i))

    if y is True:
        y_mu, y_var = m.predict_y(xy)  # Predict alpha values at test locations
        if y_a is not None:
            # plot_contourf(xx, yy, y_mu.reshape(xx.shape), a=y_a.reshape(xx.shape), contour=[0.5], title='predicted y')
            # plot_contourf_var(xx, yy, y_mu.reshape(xx.shape), y_var.reshape(xx.shape), title='predicted y')
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 0].reshape(xx.shape),
                              y_var[:, 0].reshape(xx.shape),
                              save_name='y_1_quadcopter.pdf',
                              title='predicted y dim 1')
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 1].reshape(xx.shape),
                              y_var[:, 1].reshape(xx.shape),
                              save_name='y_2_quadcopter.pdf',
                              title='predicted y dim 2')
        else:
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 0].reshape(xx.shape),
                              y_var[:, 0].reshape(xx.shape),
                              save_name='y_1_quadcopter.pdf',
                              title='predicted y dim 1')
            plot_contourf_var(xx,
                              yy,
                              y_mu[:, 1].reshape(xx.shape),
                              y_var[:, 1].reshape(xx.shape),
                              save_name='y_2_quadcopter.pdf',
                              title='predicted y dim 2')
            # plot_contourf(xx, yy, y_mu.reshape(xx.shape), y_var.reshape(xx.shape), title='predicted y')
            # TODO: how to plot variance of GPs??

    if var is True:
        lik_var, f_var = m.predict_vars(xy)
        plot_contourf_var(xx,
                          yy,
                          lik_var.reshape(xx.shape),
                          f_var.reshape(xx.shape),
                          title='noise variance vs GP covariance')


def plot_and_save_all(m, X, a_true=None):
    import datetime
    from pathlib import Path
    date = datetime.datetime.now()
    date_str = str(date.day) + "-" + str(date.month) + "/" + str(
        date.time()) + "/"
    save_dirname = '../images/model/' + date_str
    Path(save_dirname).mkdir(parents=True, exist_ok=True)

    N = 100
    x1_high = m.X.value[:, 0].max()
    x2_high = m.X.value[:, 1].max()
    x1_low = m.X.value[:, 0].min()
    x2_low = m.X.value[:, 1].min()

    xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([xx.flat, yy.flat])

    # rescale inputs
    xx = xx * X.std() + X.mean()
    yy = yy * X.std() + X.mean()
    Y = m.Y.value

    plt.quiver(X[:, 0],
               X[:, 1],
               Y[:, 0],
               np.zeros([*Y[:, 0].shape]),
               angles='xy',
               scale_units='xy',
               width=0.001,
               scale=1,
               zorder=10)
    plt.title('dx')
    plt.savefig(save_dirname + 'quiver.pdf', transparent=True)

    a_mu, a_var = m.predict_a(xy)  # Predict alpha values at test locations
    axs = plot_mean_and_var(
        xx,
        yy,
        a_mu.reshape(xx.shape),
        a_var.reshape(xx.shape),
        # a_true=a_true,
        title="alpha")
    plt.savefig(save_dirname + 'alpha.pdf', transparent=True)
    a_mu = a_mu * Y.std() + Y.mean()
    a_var = a_var * Y.std() + Y.mean()
    axs = plot_mean_and_var(
        xx,
        yy,
        a_mu.reshape(xx.shape),
        a_var.reshape(xx.shape),
        # a_true=a_true,
        title="alpha standardised")
    plt.savefig(save_dirname + 'alpha_standardised.pdf', transparent=True)

    h_mu, h_var = m.predict_h(xy)
    axs = plot_mean_and_var(
        xx,
        yy,
        h_mu.reshape(xx.shape),
        h_var.reshape(xx.shape),
        # a_true=a_true,
        title="h")
    plt.savefig(save_dirname + 'h.pdf', transparent=True)
    h_mu = h_mu * Y.std() + Y.mean()
    h_var = h_var * Y.std() + Y.mean()
    axs = plot_mean_and_var(
        xx,
        yy,
        h_mu.reshape(xx.shape),
        h_var.reshape(xx.shape),
        # a_true=a_true,
        title="h standardised")
    plt.savefig(save_dirname + 'h_standardised.pdf', transparent=True)

    f_mus, f_vars = m.predict_f(xy)
    for i, (f_mu, f_var) in enumerate(zip(f_mus, f_vars)):
        i = i + 1
        plot_mean_and_var(xx,
                          yy,
                          f_mu[:, 0].reshape(xx.shape),
                          f_var[:, 0].reshape(xx.shape),
                          title=('$f_%i$ dim 1' % i))
        plt.savefig(save_dirname + 'f' + str(i) + '_dim_1.pdf',
                    transparent=True)
        try:
            plot_mean_and_var(xx,
                              yy,
                              f_mu[:, 1].reshape(xx.shape),
                              f_var[:, 1].reshape(xx.shape),
                              title=('$f_%i$ dim 2' % i))
            plt.savefig(save_dirname + 'f' + str(i) + '_dim_2.pdf',
                        transparent=True)
        except IndexError:
            print('only one output dimension')

    y_mu, y_var = m.predict_y(xy)
    plot_mean_and_var(xx,
                      yy,
                      y_mu[:, 0].reshape(xx.shape),
                      y_var[:, 0].reshape(xx.shape),
                      title='y dim 1')
    plt.savefig(save_dirname + 'y_dim_1.pdf', transparent=True)

    try:
        plot_mean_and_var(xx,
                          yy,
                          y_mu[:, 1].reshape(xx.shape),
                          y_var[:, 1].reshape(xx.shape),
                          title='y dim 2')
        plt.savefig(save_dirname + 'y_dim_2.pdf', transparent=True)
    except IndexError:
        print('only one output dimension')

    # if var is True:
    #     lik_var, f_var = m.predict_vars(xy)
    #     plot_contourf_var(xx,
    #                       yy,
    #                       lik_var.reshape(xx.shape),
    #                       f_var.reshape(xx.shape),
    #                       title='noise variance vs GP covariance')


def plot_mean_and_var(x, y, z_mu, z_var, a=None, a_true=None, title=""):
    # fig, axs = plt.subplot(2, sharex=True, sharey=True)
    fig, axs = plt.subplots(1, 2, figsize=(24, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    surf_mu, cont_mu = plot_contour(axs[0], x, y, z_mu, a)
    cbar = fig.colorbar(surf_mu, shrink=0.5, aspect=5, ax=axs[0])
    cbar.set_label('mean')
    surf_var, cont_var = plot_contour(axs[1], x, y, z_var, a)
    cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
    cbar.set_label('variance')
    plt.suptitle(title)

    # if a_true is not None:
    #     axs[0].plot(a_true[0], a_true[1], 'k')
    #     axs[1].plot(a_true[0], a_true[1], 'k')

    return axs