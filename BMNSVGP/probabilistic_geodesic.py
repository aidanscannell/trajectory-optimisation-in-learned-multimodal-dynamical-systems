# import gpflow
# from gpflow.kernels import RBF

# kernel = gpflow.kernels.RBF(variance=2.)
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tfp.math.psd_kernels

params = np.load('saved_models/params.npz')
# l = params['l'].reshape(2, 1)
l = params['l']  # [2]
a = params['a']  # [1]
x = params['x']  # [100 x 2]

num_data = x.shape[0]
input_dim = x.shape[1]
output_dim = 1

Y = tf.constant(np.ones([num_data, output_dim]), dtype=tf.float64)

amplitude = tf.constant([a], dtype=tf.float64)
length_scale = tf.constant(l, dtype=tf.float64)
kernel = tfk.FeatureScaled(
    tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=1.),
    length_scale)
# kernel = tfk.FeatureScaled(
#     tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=length_scale),
#     length_scale)

x = tf.constant(x, dtype=tf.float64)
s = tf.constant(np.ones([1, input_dim]), dtype=tf.float64)

Kxx = kernel.matrix(x, x)
jitter = 1e-8
diagonal = tf.linalg.diag(
    tf.constant(jitter * np.ones([Kxx.shape[-1]]), dtype=tf.float64))
Kxx = Kxx + jitter
iKxx = tf.linalg.inv(Kxx)

with tf.GradientTape(persistent=True) as g:
    g.watch(s)
    with tf.GradientTape(persistent=True) as t:
        t.watch(s)
        Kxs = kernel.matrix(x, s)
        with tf.GradientTape() as tt:
            tt.watch(s)
            Kss = kernel.matrix(s, s)
        dk_dss = tt.jacobian(Kss, s)

    dk_ds = t.jacobian(Kxs, s)
    dk_ds = tf.squeeze(dk_ds)

    d2k_d2s = t.jacobian(dk_dss, s)

    d2k_d2s = tf.squeeze(d2k_d2s)

    # calculate mean and variance of J
    dk_dsT = tf.transpose(dk_ds)
    mu_j = dk_dsT @ iKxx @ Y  # [input_dim x 1]
    assert mu_j.shape == [1, input_dim, 1]
    cov_j = d2k_d2s - dk_dsT @ iKxx @ dk_ds  # [input_dim x inut_dim]
    assert cov_j.shape == [1, input_dim, input_dim]

    mu_jT = tf.linalg.matrix_transpose(mu_j)
    assert mu_jT.shape == [1, 1, input_dim]

    jTj = mu_jT @ mu_j  # [1 x 1]
    assert jTj.shape == [1, 1, 1]

    G = jTj + output_dim * cov_j  # [input_dim x input_dim]
    assert G.shape == [1, input_dim, input_dim]
    vecG = tf.reshape(G, [-1])  # stack columns of G
    assert vecG.shape == [input_dim * input_dim]

dGds = g.jacobian(vecG, s)
assert dGds.shape == [input_dim * input_dim, 1, input_dim]
dGds = tf.squeeze(dGds)
assert dGds.shape == [input_dim * input_dim, input_dim]

# vecG = tf.stack([G[0, :], G[1, :]])  # [input_dim^2 x input_dim]
# print(vecG.shape)

# with tf.GradientTape() as t:
#     t.watch(s)
#     Kxs = kernel.matrix(x, s)
# dk_ds = t.jacobian(Kxs, s)
# dk_ds = tf.squeeze(dk_ds)

# with tf.GradientTape() as t:
#     t.watch(s)
#     with tf.GradientTape() as tt:
#         tt.watch(s)
#         Kss = kernel.matrix(s, s)
#     dk_dss = tt.jacobian(Kss, s)
# d2k_d2s = t.jacobian(dk_dss, s)

# d2k_d2s = tf.squeeze(d2k_d2s)

# # calculate mean and variance of J
# iKxx = tf.linalg.inv(Kxx)
# dk_dsT = tf.transpose(dk_ds)
# mu_j = dk_dsT @ iKxx @ Y  # [input_dim x 1]
# cov_j = d2k_d2s - dk_dsT @ iKxx @ dk_ds  # [input_dim x inut_dim]

# mu_jT = tf.transpose(mu_j)
# jTj = mu_jT @ mu_j  # [1 x 1]

# with tf.GradientTape() as t:
#     t.watch(s)
#     G = jTj + output_dim * cov_j  # [input_dim x input_dim]

# dGds = t.gradient(G, s)
# print(dGds)
# # vecG = tf.stack([G[0, :], G[1, :]])  # [input_dim^2 x input_dim]
# # print(vecG.shape)
