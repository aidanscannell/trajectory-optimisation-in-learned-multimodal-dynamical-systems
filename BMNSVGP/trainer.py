import tensorflow as tf
import gpflow
from utils import run_adam, plot_loss, plot_model, plot_a
from model import BMNSVGP
from gen_data import gen_data

float_type = tf.float64

X, Y, a = gen_data(600,
                   frac=0.5,
                   low_noise_var=0.005,
                   high_noise_var=0.3,
                   plot_flag=False)
X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()

N = X.shape[0]
D = X.shape[1]
F = Y.shape[1]

print('before model ')
gpflow.reset_default_graph_and_session()

with gpflow.defer_build():
    m = BMNSVGP(X, Y, var_low=0.005, var_high=0.3, minibatch_size=100)
#     m = BMNSVGP(X, Y, var_low=0.001, var_high=0.7, minibatch_size=100)
m.compile()

print('Optimising model...')

logger = run_adam(m, maxiter=gpflow.test_util.notebook_niter(15000))

print('Finished optimising model.')

plot_loss(logger)
plot_model(m, m1=True, m2=True)
plot_a(m, a)
plot_model(m, a=True)
plot_model(m, y=True)

# saver = gpflow.saver.Saver()
# saver.save('../saved_models/bmnsvgp', m)
