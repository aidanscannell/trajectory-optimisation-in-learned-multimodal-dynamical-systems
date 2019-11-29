import matplotlib.pyplot as plt
import numpy as np

x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 10
y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 10

data = np.load('../npz/model_inputs_combined.npz')
X = data['x']
Y = data['y']
print(X)
plt.figure()
plt.quiver(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1])
plt.plot(x_alpha, y_alpha)
plt.savefig('quiver.pdf', transparent=True)
plt.show(block=True)
