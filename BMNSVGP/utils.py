import gpflow
import matplotlib.pyplot as plt
import numpy as np


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


def plot_model(m,
               m1=False,
               m2=False,
               a=False,
               h=False,
               y=False,
               save_name=False):
    fig = plt.figure(figsize=(12, 4))
    pX = np.linspace(-1, 1, 100)[:, None]  # Test locations

    plt.plot(m.X.value,
             m.Y.value,
             'x',
             color='k',
             label='Observations',
             alpha=0.2)
    if m1 or m2 or a or h:
        #     plt.plot(m.feature_f_low.Z.value, np.zeros(m.feature_f_low.Z.value.shape), 'k|', mew=2, label='Inducing locations mode 1')
        plt.plot(m.feature_f_low.Z.value,
                 np.zeros(m.feature_f_low.Z.value.shape),
                 'k|',
                 mew=2)
        plt.plot(m.feature_f_high.Z.value,
                 np.zeros(m.feature_f_high.Z.value.shape),
                 'b|',
                 mew=2)
    #     plt.plot(m.feature_f_high.Z.value, np.zeros(m.feature_f_high.Z.value.shape), 'b|', mew=2, label='Inducing locations mode 2')
    #     plt.plot(m.feature_f_low.Z.value, np.zeros(m.feature_f_low.Z.value.shape), 'k|', mew=2, label='Inducing locations mode 1')

    if h is True:
        a_mu, a_var = m.predict_h(pX)  # Predict alpha values at test locations
        plt.plot(pX, a_mu, color='olive', lw=1.5)
        plt.fill_between(pX[:, 0], (a_mu - 2 * a_var**0.5)[:, 0],
                         (a_mu + 2 * a_var**0.5)[:, 0],
                         color='olive',
                         alpha=0.4,
                         lw=1.5,
                         label='Separation manifold GP')

    if a is True:
        a_mu, a_var = m.predict_a(pX)  # Predict alpha values at test locations
        plt.plot(pX, a_mu, color='olive', lw=1.5)
        plt.fill_between(pX[:, 0], (a_mu - 2 * a_var**0.5)[:, 0],
                         (a_mu + 2 * a_var**0.5)[:, 0],
                         color='blue',
                         alpha=0.4,
                         lw=1.5,
                         label='$\\alpha$')

    if m1 is True:
        pY_low, pYv_low = m.predict_f_high(pX)
        #         pY_low, pYv_low = m.predict_f_low(pX)
        line, = plt.plot(pX, pY_low, color='m', alpha=0.6, lw=1.5)
        plt.fill_between(pX[:, 0], (pY_low - 2 * pYv_low**0.5)[:, 0],
                         (pY_low + 2 * pYv_low**0.5)[:, 0],
                         color='m',
                         alpha=0.2,
                         lw=1.5,
                         label='Mode 1 - low noise')

    if m2 is True:
        pY_high, pYv_high = m.predict_f_low(pX)
        #         pY_high, pYv_high = m.predict_f_high(pX)
        line, = plt.plot(pX, pY_high, color='c', alpha=0.6, lw=1.5)
        plt.fill_between(pX[:, 0], (pY_high - 2 * pYv_high**0.5)[:, 0],
                         (pY_high + 2 * pYv_high**0.5)[:, 0],
                         color='c',
                         alpha=0.2,
                         lw=1.5,
                         label='Mode 2 - high noise')

    if y is True:
        pY, pYv = m.predict_y(pX)
        line, = plt.plot(pX, pY, color='royalblue', alpha=0.6, lw=1.5)
        plt.fill_between(pX[:, 0], (pY - 2 * pYv**0.5)[:, 0],
                         (pY + 2 * pYv**0.5)[:, 0],
                         color='royalblue',
                         alpha=0.2,
                         lw=1.5)


#         plt.fill_between(pX[:, 0], (pY-2*pYv**0.5)[:, 0], (pY+2*pYv**0.5)[:, 0], color='royalblue', alpha=0.2, lw=1.5, label='Combined')

#     pY, pYv = m.predict_y1(pX)
#     line, = fig.plot(pX, pY, color='red', alpha=0.6, lw=1.5)
#     fig.fill_between(pX[:, 0], (pY-2*pYv**0.5)[:, 0], (pY+2*pYv**0.5)[:, 0], color='red', alpha=0.2, lw=1.5, label='Combined2')

    fig.legend(loc='lower right', fontsize=15)
    plt.xlabel('$(\mathbf{s}_{t-1}, \mathbf{a}_{t-1})$', fontsize=30)
    plt.ylabel('$\mathbf{s}_t$', fontsize=30)
    #     plt.xlim(-1.0, 1.2)
    plt.tick_params(labelsize=20)
    if save_name is not False:
        plt.savefig(save_name, transparent=True, bbox_inches='tight')
