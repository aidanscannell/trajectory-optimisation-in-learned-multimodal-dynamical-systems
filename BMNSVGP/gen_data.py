import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.sin(
        x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)


def gen_data(N=600,
             frac=0.4,
             low_noise_var=0.005,
             high_noise_var=0.3,
             plot_flag=False):
    """
    N - number of training observations
    a - fraction of observations in mode 2 (high noise mode)
    """
    N2 = int(frac * N / 2)  # number of observations in mode 2 (high noise)
    # generate input observations X
    #     X = rnd.randn(N, 1) * 2 - 1 # X values
    # X = np.random.rand(N, 1) * 2 - 1  # X values
    X = np.linspace(-1.0, 1.0, N)[:, None]

    # generate target observations Y
    # Y = np.zeros([N, 1])
    Y = func(X)  # + 3
    Y[X > 0] += low_noise_var * np.random.randn((Y[X > 0]).shape[0])
    Y[X < 0] += high_noise_var * np.random.randn(
        (Y[X < 0]).shape[0])  # add noise to subset of target observations Y
    #     Y[N2:-N2] += low_noise_var * rnd.randn(N-2*N2,1)
    #     Y[-N2:] += high_noise_var  * rnd.randn(N2,1) # add noise to subset of target observations Y
    #     Y[:N2] += high_noise_var* rnd.randn(N2,1) # add noise to subset of target observations Y

    # Bernoulli indicator variable, 0 = low noise, 1 = high noise
    a = np.zeros([N, 1])
    a[X < 0] = 1
    # a[-N2:] = 1
    # a[:N2] = 1

    if plot_flag is True:
        plot_data(X, Y, a, N2, func)

    return X, Y, a


def plot_data(X, Y, a, N2, func, title=None):
    plt.figure(figsize=(12, 4))
    plt.plot(X, Y, 'x', color='k', alpha=0.4, label="Observations")

    #     plt.fill_between(X[:N2, 0], -1.8, 1.8, color='c', alpha=0.2, lw=1.5)
    plt.fill_between(X[a == 0],
                     -1.8,
                     1.8,
                     color='m',
                     alpha=0.2,
                     lw=1.5,
                     label='Mode 1 (low noise)')
    plt.fill_between(X[-N2:, 0],
                     -1.8,
                     1.8,
                     color='c',
                     alpha=0.2,
                     lw=1.5,
                     label='Mode 2 (high noise)')
    #     Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
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
