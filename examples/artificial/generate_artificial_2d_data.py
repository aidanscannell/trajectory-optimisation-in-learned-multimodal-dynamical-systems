import matplotlib.pyplot as plt
import numpy as np
from ProbGeo.visualisation.gp import plot_contourf
from scipy.stats import multivariate_normal

filename = "./data/artificial_data_2d.npz"

# generate data set
input_dim = 2
N = 961  # number of training observations
N = 1600  # number of training observations

low_lim = -3
high_lim = 3
# TODO: higher order root depending on input_dim
sqrtN = int(np.sqrt(N))
X = np.sort(
    np.random.rand(sqrtN, input_dim) * (high_lim - low_lim) + low_lim, axis=0
)
x, y = np.meshgrid(X[:, 0], X[:, 1])
X = np.column_stack([x.flat, y.flat])  # Need an (N, 2) array - N (x, y) pairs.

# Generate alpha (separation) from a mixture of gaussians
num_mixtures = 2
means = np.array([[1.5, -1.0], [1.5, -1.0]])
covs = np.random.rand(num_mixtures, input_dim, input_dim) * 0.5
covs = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.1, 0.0], [0.0, 0.1]]])
covs = [np.diag(np.diag(covs[i, :, :])) for i in range(num_mixtures)]
a = sum(
    [
        multivariate_normal.pdf(X, mean=mean, cov=cov)
        for mean, cov in zip(means, covs)
    ]
).reshape(N, 1)
a = np.interp(a, (a.min(), a.max()), (0, +1))  # rescale to range 0, 1

# remove some data points
x1_low = -1.5
x2_low = -0.5
x1_high = 0.0
x2_high = 2.0
mask_0 = X[:, 0] < x1_low
mask_1 = X[:, 1] < x2_low
mask_2 = X[:, 0] > x1_high
mask_3 = X[:, 1] > x2_high
mask = mask_0 | mask_1 | mask_2 | mask_3
X_partial = X[mask, :]
Y_partial = a[mask, :]

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plot_contourf(fig, ax, x, y, a.reshape(sqrtN, sqrtN))
plt.scatter(X_partial[:, 0], X_partial[:, 1], color="k", marker="x")
plt.show()

np.savez(filename, x=X_partial, y=Y_partial)
