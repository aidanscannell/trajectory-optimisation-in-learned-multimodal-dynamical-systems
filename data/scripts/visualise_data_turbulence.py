import matplotlib.pyplot as plt
import numpy as np

x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 1000
y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 1000

# data = np.load('../npz/model_inputs_combined.npz')
# data = np.load('../npz/model_data.npz')
# data = np.load('../npz/model_data_vel.npz')
data = np.load('../npz/turbulence/model_data.npz')
X = data['x']
Y = data['y']
print(X.shape)
plt.figure()
print(Y.shape)
# YY = np.sqrt(Y[:, 0]**2 + Y[:, 1]**2 + Y[:, 2]**2)
# YYY = np.zeros(YY.shape)
# plt.quiver(X[:, 0], X[:, 1], YY, YYY)
dxyz = Y[:, 3]
x = X[:, 0]
y = X[:, 1]
fig, ax2 = plt.subplots(nrows=1)
ax2.tricontour(x, y, dxyz, levels=14, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x, y, dxyz, levels=14, cmap="RdBu_r")
plt.scatter(x, y, color='k', marker='x')
plt.show()
