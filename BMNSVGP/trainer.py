import gpflow
import numpy as np
import tensorflow as tf

from gen_data import gen_data
from gen_data_2d import gen_data_2d
from model import BMNSVGP
from utils import plot_a, plot_contourf, plot_loss, plot_model, run_adam

float_type = tf.float64

# generate 1D input
# X, Y, a = gen_data(600,
#                    frac=0.5,
#                    low_noise_var=0.005,
#                    high_noise_var=0.3,
#                    plot_flag=False)

# generate 2D input
N = 1000  # number of training observations
X, Y, a = gen_data_2d(N=N,
                      num_mixtures=5,
                      x_scale=2.,
                      y_scale=2.,
                      plot_flag=[1, 1, 1])

X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()

N = X.shape[0]
D = X.shape[1]
F = Y.shape[1]

print('before model ')
gpflow.reset_default_graph_and_session()

var_low = np.array([[[0.005, 0.], [0., 0.005]]])
var_high = np.array([[[0.3, 0.], [0., 0.3]]])

with gpflow.defer_build():
    m = BMNSVGP(X, Y, var_low=var_low, var_high=var_high, minibatch_size=100)
#     m = BMNSVGP(X, Y, var_low=0.001, var_high=0.7, minibatch_size=100)
m.compile()

print('Optimising model...')

logger = run_adam(m, maxiter=gpflow.test_util.notebook_niter(15000))

print('Finished optimising model.')

plot_loss(logger)
# plot_model(m, m1=True, m2=True)
# plot_a(m, a)
# plot_model(m, a=True)
# plot_model(m, y=True)

# saver = gpflow.saver.Saver()
# saver.save('../saved_models/bmnsvgp', m)
