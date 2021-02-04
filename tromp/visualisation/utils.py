import jax.numpy as jnp


def create_grid(X, N):
    # num_dims = X.shape[-1]
    # num_data = X.shape[-2]
    x1_low, x1_high, x2_low, x2_high = (
        X[:, 0].min(),
        X[:, 0].max(),
        X[:, 1].min(),
        X[:, 1].max(),
    )
    x1_low *= 1.3
    x2_low *= 1.3
    x1_high *= 1.3
    x2_high *= 1.3
    # x2_low = -3.0
    # x1_low *= 1.5
    # x2_low *= 1.5
    # x1_high *= 1.5
    # x2_high *= 1.5
    # mins = X.min(axis=0)
    # print('mins')
    # print(mins)
    # print(mins.shape)
    # maxs = X.max(axis=0)
    # print(maxs)
    # print(maxs.shape)

    sqrtN = int(jnp.sqrt(N))
    xx = jnp.linspace(x1_low, x1_high, sqrtN)
    yy = jnp.linspace(x2_low, x2_high, sqrtN)
    xx, yy = jnp.meshgrid(xx, yy)
    xy = jnp.column_stack([xx.reshape(-1), yy.reshape(-1)])
    return xy, xx, yy
    # return jnp.array(xy)
