# import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from matplotlib import cm
from scipy.integrate import solve_bvp

from geodesic_func import geodesic_fun

# from utils import plot_contourf


def fun(t, y):
    c = np.vstack((y[0], y[1])).T
    g = np.vstack((y[2], y[3])).T
    dydt = vmap(geodesic_fun, in_axes=(0))(c, g)  # [10 x 2 x 1]
    return np.vstack((y[2], y[3], dydt[:, 0, 0], dydt[:, 1, 0]))  # [4 x 10]


# def bc(ya, yb):
#     return np.array([ya[0], ya[1], yb[0], yb[1]])


def bc(ya, yb):
    return np.array([ya[0], ya[1], None, None])


start = [0., -1.6]
end = [0., 1.2]
m = 10
input_dim = 2
t = np.linspace(0, 1, m)
# y = np.zeros((input_dim, m))
# start = [0., -1.6]
# end = [0., 1.2]
x1 = np.linspace(0, -1.6, m)
x2 = np.linspace(0, 1.2, m)
# dx1 = np.linspace(0., 0., m)
# dx2 = np.linspace(0., 0., m)
dx1 = np.ones(m) * 0.03
dx2 = np.ones(m) * 0.01
y = np.vstack([x1, x2, dx1, dx2])

sol = solve_bvp(fun, bc, t, y, verbose=2)
print(sol)
x1 = sol.y[0, :]
x2 = sol.y[1, :]
# plt.plot(x1, x2)

filename = 'saved_models/params.npz'
params = np.load(filename)
lengthscale = params['l']  # [2]
var = params['var']  # [1]
X = params['x']  # [num_data x 2]
y = params['a']  # [num_data x 2] meen and var of alpha
Y = y[:, :, 0].T  # [num_data x 1]

N = 961  # number of training observations


def plot_contour(ax, X, z, N):
    x1_high = X[:, 0].max()
    x2_high = X[:, 1].max()
    x1_low = X[:, 0].min()
    x2_low = X[:, 1].min()
    N = np.sqrt(N)
    xx, yy = np.mgrid[x1_low:x1_high:N * 1j, x2_low:x2_high:N * 1j]
    surf = ax.contourf(xx,
                       yy,
                       z.reshape(xx.shape),
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
    return surf


fig, axs = plt.subplots(1, 2, figsize=(24, 4))
plt.subplots_adjust(wspace=0, hspace=0)
surf_mean = plot_contour(axs[0], X, Y[:, 0:1], N)
surf_var = plot_contour(axs[1], X, Y[:, 1:2], N)
cbar = fig.colorbar(surf_mean, shrink=0.5, aspect=5, ax=axs[0])
cbar.set_label('mean')
cbar = fig.colorbar(surf_var, shrink=0.5, aspect=5, ax=axs[1])
cbar.set_label('variance')

# plt.plot(x1, x2)
for ax in axs:
    ax.plot(x1, x2, 'k')
    ax.scatter(start[0], start[1], color='k', marker='x')
    ax.annotate("start", (start[0], start[1]))
    ax.scatter(end[0], end[1], color='k', marker='x')
    ax.annotate("end", (end[0], end[1]))

plt.show()

# x_plot = np.linspace(0, 1, 100)
# y_plot = sol.sol(x_plot)[0]
# plt.plot(x_plot, y_plot)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
