import matplotlib.pyplot as plt
import numpy as np

x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 1000
y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 1000

# data = np.load('../npz/model_inputs_combined.npz')
# data = np.load('../npz/model_data.npz')
data = np.load('../npz/model_data_vel.npz')
X = data['x']
Y = data['y']
print(X)
plt.figure()
print(Y.shape)
YY = np.sqrt(Y[:, 0]**2 + Y[:, 1]**2 + Y[:, 2]**2)
YYY = np.zeros(YY.shape)
# plt.quiver(X[:, 0], X[:, 1], YY, YYY)
plt.quiver(X[:, 0], X[:, 1], Y[:, 3], YYY)
# plt.quiver(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1])
plt.plot(x_alpha, y_alpha)
plt.savefig('quiver_vel.pdf', transparent=True)
plt.show(block=True)
