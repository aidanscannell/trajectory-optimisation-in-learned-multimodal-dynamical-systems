import matplotlib.pyplot as plt
import numpy as np

data = np.load(
    '../npz/turbulence/model_data_fan_fixed_middle_short_lengthscale.npz')
X = data['x']
Y = data['y']
print(X.shape)
print(Y.shape)
Y0 = np.zeros(Y.shape)
plt.quiver(X[:, 0], X[:, 1], Y, Y0)
dxyz = Y[:, 3]
x = X[:, 0]
y = X[:, 1]
fig, ax2 = plt.subplots(nrows=1)
ax2.tricontour(x, y, dxyz, levels=14, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x, y, dxyz, levels=14, cmap="RdBu_r")
plt.scatter(x, y, color='k', marker='x')
plt.show()
