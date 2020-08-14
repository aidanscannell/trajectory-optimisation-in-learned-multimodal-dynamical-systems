import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gpflow
import tensorflow as tf

from gpflow import settings
from utils import plot_contourf, plot_model_2d_uav

# x1_high = X[:, 0].max()
# x2_high = X[:, 1].max()
# x1_low = X[:, 0].min()
# x2_low = X[:, 1].min()
# N = np.sqrt(N)
# xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
# xy = np.column_stack([xx.flat, yy.flat])

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
filename = "../saved_models/model_feb12"
saver = gpflow.saver.Saver()
m = saver.load(filename)

# print(m.as_pandas_table())
# print(m.X)
X = m.X.value
lengthscales = m.read_trainables()['BMNSVGP/kern_h/lengthscales']
variance = m.read_trainables()['BMNSVGP/kern_h/variance']

# print(lengthscales)
# print(variance)
# print(X.shape)

m.compile()
alpha = m.predict_a(X)  # mean and var of alpha

np.savez('../saved_models/params_from_model',
         l=lengthscales,
         var=variance,
         x=X,
         a=alpha)

print(X.shape)
print(alpha.shape)
