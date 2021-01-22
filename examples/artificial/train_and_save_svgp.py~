import GPy
import numpy as np
from matplotlib import pyplot as plt
# import climin

from ProbGeo.visualisation.gp import plot_mean_and_var
from ProbGeo.visualisation.utils import create_grid

filename = '../../data/processed/artificial_data_2d.npz'
data = np.load(filename)
X = data['x']
Y = data['y']
print(X.shape)
print(Y.shape)

num_data = X.shape[0]
input_dim = X.shape[1]
output_dim = Y.shape[1]

mean_func = GPy.mappings.Constant(input_dim=input_dim, output_dim=output_dim)
lik = GPy.likelihoods.Gaussian()
kern = GPy.kern.RBF(input_dim=input_dim, lengthscale=1., ARD=True)

M = 400
idx = np.random.choice(range(num_data), size=M, replace=False)
Z = X[idx, ...].reshape(-1, input_dim)
print(Z.shape)
batchsize = 70
m = GPy.core.SVGP(X,
                  Y,
                  Z,
                  likelihood=lik,
                  mean_function=mean_func,
                  kernel=kern,
                  batchsize=batchsize)
# batchsize=batchsize)
# m.kern.white.variance = 1e-5
m.Gaussian_noise.variance = 1e-5
m.Gaussian_noise.variance.fix()
# m.kern.white.fix()
m.Z.fix()

print(m)

Xnew, xx, yy = create_grid(X, N=961)
mu, var = m.predict(Xnew)
fig, axs = plot_mean_and_var(xx, yy, mu, var)
plt.show()

opt = m.optimize(messages=True, max_iters=2000000)
print(opt)
print(m)
mu, var = m.predict(Xnew)
fig, axs = plot_mean_and_var(xx, yy, mu, var)
plt.show()
