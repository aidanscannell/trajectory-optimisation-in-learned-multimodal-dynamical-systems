import gpflow
import numpy as np
import tensorflow as tf
from gpflow import settings

from derivative_kernel import SquaredExponentialDerivative
from utils import plot_model_2d_uav

filename = './saved_models/fake_alpha_for_opt'
# with tf.Graph().as_default() as graph, tf.Session().as_default():

session = gpflow.get_default_session()
saver = gpflow.saver.Saver()
m = saver.load(filename)

# print(m.read_trainables())
# print(m.read_trainables()['BMNSVGP/kern_h/lengthscales'])
# print(m.read_trainables()['BMNSVGP/kern_h/variance'])
# kern_h = m.kern_h
lengthscales = m.read_trainables()['BMNSVGP/kern_h/lengthscales']
variance = m.read_trainables()['BMNSVGP/kern_h/variance']
kern_h = SquaredExponentialDerivative(input_dim=2,
                                      lengthscales=lengthscales,
                                      variance=variance)
print(kern_h)

x_star = tf.placeholder(settings.float_type, shape=(1, 2))
X = tf.constant(m.X.value)

dk = kern_h.dKdX(X, x_star)
d2k = kern_h.d2Kd2X(x_star, x_star)

kern_h.initialize(session)

x_star_ = np.random.rand(1, 2)
dK_ = session.run(dk, feed_dict={x_star: x_star_})
d2K_ = session.run(d2k, feed_dict={x_star: x_star_})
print(dK_.shape)
print(d2K_.shape)
