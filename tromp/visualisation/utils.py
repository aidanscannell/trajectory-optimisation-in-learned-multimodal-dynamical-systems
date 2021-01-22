import numpy as np


def create_grid(X, N):
    x1_low, x1_high, x2_low, x2_high = X[:, 0].min(), X[:,
                                                        0].max(), X[:, 1].min(
                                                        ), X[:, 1].max()
    # x1_low, x1_high, x2_low, x2_high = -2., 3., -3., 3.
    x1_low *= 1.3
    x2_low *= 1.3
    x1_high *= 1.3
    x2_high *= 1.3
    x2_low = -3.
    # x1_low *= 1.5
    # x2_low *= 1.5
    # x1_high *= 1.5
    # x2_high *= 1.5
    sqrtN = int(np.sqrt(N))
    xx = np.linspace(x1_low, x1_high, sqrtN)
    yy = np.linspace(x2_low, x2_high, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    return xy, xx, yy
