import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tfp.math.psd_kernels

# Load kernel hyper-params and create tf kernel
params = np.load('saved_models/params.npz')
l = params['l']  # [2]
a = params['var']  # [1]
x = params['x']  # [num_data x 2]
y = params['a']  # [num_data x 2] meen and var of alpha
y = y[0:1, :, 0]
x = tf.constant(x, dtype=tf.float64)
Y = tf.constant(y.T, dtype=tf.float64)
amplitude = tf.constant(a, dtype=tf.float64)
length_scale = tf.constant(l, dtype=tf.float64)
kernel = tfk.FeatureScaled(
    tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=1.),
    length_scale)

num_data, input_dim, output_dim = x.shape[0], x.shape[1], 1

Kxx = kernel.matrix(x, x)
jitter = 1e-8
diagonal = tf.linalg.diag(
    tf.constant(jitter * np.ones([Kxx.shape[-1]]), dtype=tf.float64))
Kxx = Kxx + jitter
iKxx = tf.linalg.inv(Kxx)


def ode_fn(t, y):
    c = y[0]
    g = y[1]
    assert c.shape == [1, input_dim]
    assert g.shape == [1, input_dim]

    with tf.GradientTape(persistent=True) as grad:
        grad.watch(c)
        with tf.GradientTape(persistent=True) as t:
            t.watch(c)
            Kxs = kernel.matrix(x, c)
            with tf.GradientTape() as tt:
                tt.watch(c)
                Kss = kernel.matrix(c, c)
            dk_dss = tt.jacobian(Kss, c)

        dk_ds = t.jacobian(Kxs, c)
        dk_ds = tf.squeeze(dk_ds)

        d2k_d2s = t.jacobian(dk_dss, c)

        d2k_d2s = tf.squeeze(d2k_d2s)

        # calculate mean and variance of J
        dk_dsT = tf.transpose(dk_ds)  # [input_dim x num_data]
        mu_j = dk_dsT @ iKxx @ Y  # [input_dim x 1]
        assert mu_j.shape == [input_dim, 1]
        # assert mu_j.shape == [1, input_dim, 1]
        cov_j = d2k_d2s - dk_dsT @ iKxx @ dk_ds  # [input_dim x inut_dim]
        assert cov_j.shape == [input_dim, input_dim]
        # assert cov_j.shape == [1, input_dim, input_dim]

        mu_jT = tf.linalg.matrix_transpose(mu_j)
        assert mu_jT.shape == [1, input_dim]
        # assert mu_jT.shape == [1, 1, input_dim]

        # TODO make this DxD
        jTj = mu_j @ mu_jT  # [D x D]
        # jTj = mu_jT @ mu_j  # [1 x 1]
        # assert jTj.shape == [1, 1, 1]
        assert jTj.shape == [input_dim, input_dim]

        G = jTj + output_dim * cov_j  # [input_dim x input_dim]
        assert G.shape == [input_dim, input_dim]
        # assert G.shape == [1, input_dim, input_dim]
        vecG = tf.reshape(G, [-1])  # stack columns of G
        assert vecG.shape == [input_dim * input_dim]

    dGds = grad.jacobian(vecG, c)
    assert dGds.shape == [input_dim * input_dim, 1, input_dim]
    dGds = tf.squeeze(dGds)
    assert dGds.shape == [input_dim * input_dim, input_dim]
    dGdsT = tf.transpose(dGds)
    assert dGdsT.shape == [input_dim, input_dim * input_dim]

    invG = tf.linalg.inv(G)

    gT = tf.transpose(g)
    assert gT.shape == [input_dim, 1]
    operator_gT = tf.linalg.LinearOperatorFullMatrix(gT)
    CkronC = tf.linalg.LinearOperatorKronecker([operator_gT, operator_gT])
    assert CkronC.shape == [input_dim * input_dim, 1]
    dg = -0.5 * invG @ dGdsT @ CkronC.to_dense()
    dgT = tf.transpose(dg)
    assert dgT.shape == [1, input_dim]
    dc = g
    assert dc.shape == [1, input_dim]
    y_out = [dc, dgT]
    # return y_out
    return [dc, dgT]


# t_init, t0, t1 = 0., 0.5, 1.
# t_init = tf.constant([0., 0.], dtype=tf.float64)
t_init = 0.
t0 = 0.1
t1 = 1.
# solution_times = tf.constant(np.linspace(t0, t1, 3), dtype=tf.float64)
# solution_times = [0.4, 0.8, 1.]
solution_times = [0.1, 0.2]
# y_init = tf.constant([1., 1.], dtype=tf.float64)

s = tf.constant(np.ones([1, input_dim]), dtype=tf.float64)
# dg = ode_fn(s, [s, s])

c = tf.constant(np.ones([1, input_dim]), dtype=tf.float64) * 0.
g = tf.constant(np.ones([1, input_dim]), dtype=tf.float64) * 0.1
y_init = [c, g]
results = tfp.math.ode.BDF().solve(ode_fn,
                                   t_init,
                                   y_init,
                                   solution_times=solution_times)
y0 = results.states[0]  # == dot(matrix_exp(A * t0), y_init)
y1 = results.states[1]  # == dot(matrix_exp(A * t1), y_init)
with tf.Session() as sess:
    # results = sess.run(dg)
    results = sess.run(results)
    print(results.states)
    # print(y0)
    # print(y1)
