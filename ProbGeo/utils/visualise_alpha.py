import matplotlib.pyplot as plt
import numpy as np

from utils import plot_contourf

filename = '../saved_models/params.npz'
params = np.load(filename)
lengthscale = params['l']  # [2]
var = params['var']  # [1]
X = params['x']  # [num_data x 2]
y = params['a']  # [num_data x 2] meen and var of alpha
Y = y[0:1, :, 0].T  # [num_data x 1]

N = 961  # number of training observations

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

plot_contourf(xx,
              yy,
              Y[:, 0:1].reshape(xx.shape),
              a=None,
              contour=None,
              title=None,
              save_name=None)
start = [0., -1.6]
end = [0., 1.2]
