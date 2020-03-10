import numpy as np

import gpflow
import tensorflow as tf
from utils.gen_data import func, func1, func2, gen_data
from gpflow import settings

from trainer import train_model, save_model
from utils.utils import plot_contourf, plot_model_2d_uav

# generate data set
input_dim = 2
output_dim = 2
func_list = [func1, func2]
N = 961  # number of training observations
X, Y, a = gen_data(N,
                   output_dim,
                   low_lim=-2.,
                   high_lim=2.,
                   func_list=func_list,
                   plot_flag=False)

# remove some data points
mask_0 = X[:, 0] < 0
mask_1 = X[:, 1] > -0
mask = mask_0 | mask_1
X_partial = X[mask, :]
Y_partial = Y[mask, :]

print(Y.shape)

x1_high = X[:, 0].max()
x2_high = X[:, 1].max()
x1_low = X[:, 0].min()
x2_low = X[:, 1].min()
N = np.sqrt(N)
xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
xy = np.column_stack([xx.flat, yy.flat])
# rescale inputs
# xx = xx * X.std() + X.mean()
# yy = yy * X.std() + X.mean()

# plot each function
# for i in range(Y.shape[1]):
#     plot_contourf(xx,
#                   yy,
#                   Y[:, i:i + 1].reshape(xx.shape),
#                   a=None,
#                   contour=None,
#                   title=None,
#                   save_name=None)

m, logger = train_model(X,
                        Y,
                        num_iters=15000,
                        vars_list=None,
                        minibatch_size=100)
a_true = None
X_missing = None
save_model(m, X, a_true, X_missing)

# alpha = m.predict_a(X)  # mean and var of alpha
# lengthscales = m.kern_h.lengthscales.value
# variance = m.kern_h.variance.value

# np.savez('../saved_models/params_fake_sparse', l=lengthscales, var=variance, x=X, a=alpha)
# #  filename = './saved_models/fake_alpha_for_opt'

# # session = gpflow.get_default_session()
# saver = gpflow.saver.Saver()
# m = saver.load(filename)

# lengthscales = m.read_trainables()['BMNSVGP/kern_h/lengthscales']
# variance = m.read_trainables()['BMNSVGP/kern_h/variance']
# X = m.X.value
# # Y = m.Y.value
# print(X.shape)
# # print(Y.shape)
# a = m.predict_a(X)
# print(a.shape)

# np.savez('saved_models/params', l=lengthscales, a=variance, x=X, alpha=a)
