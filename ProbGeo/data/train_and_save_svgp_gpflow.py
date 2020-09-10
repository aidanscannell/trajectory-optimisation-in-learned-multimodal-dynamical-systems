import gpflow as gpf
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from ProbGeo.visualisation.gp import plot_mean_and_var
from ProbGeo.visualisation.utils import create_grid

load_data_filename = '../../data/processed/artificial_data_2d.npz'
save_params_filename = '../../models/saved_models/params_fake_sparse_26-08'

batch_size = 400
epochs = 1500
logging_epoch_freq = 100
num_inducing = 300

data = np.load(load_data_filename)
X = data['x']
Y = data['y']
print(X.shape)
print(Y.shape)
dataset = (X, Y)

num_data = X.shape[0]
input_dim = X.shape[1]
output_dim = Y.shape[1]

mean_func = gpf.mean_functions.Constant()
lik = gpf.likelihoods.Gaussian(2e-6)
lengthscales = np.array([1, 1])
kern = gpf.kernels.RBF(lengthscales=lengthscales)

idx = np.random.choice(range(num_data), size=num_inducing, replace=False)
inducing_variable = X[idx, ...].reshape(-1, input_dim)
inducing_variable = gpf.inducing_variables.InducingPoints(inducing_variable)

m = gpf.models.SVGP(
    kernel=kern,
    likelihood=lik,
    inducing_variable=inducing_variable,
    mean_function=mean_func,
)

gpf.utilities.print_summary(m)
gpf.set_trainable(m.likelihood.variance, False)
gpf.set_trainable(m.inducing_variable, False)
gpf.utilities.print_summary(m)

Xnew, xx, yy = create_grid(X, N=961)
# mu, var = m.predict_y(Xnew)
# fig, axs = plot_mean_and_var(xx, yy, mu.numpy(), var.numpy())
# plt.show()

optimizer = tf.optimizers.Adam()

prefetch_size = tf.data.experimental.AUTOTUNE
shuffle_buffer_size = num_data // 2
num_batches_per_epoch = num_data // batch_size
train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
train_dataset = (train_dataset.repeat().prefetch(prefetch_size).shuffle(
    buffer_size=shuffle_buffer_size).batch(batch_size, drop_remainder=True))
training_loss = m.training_loss_closure(iter(train_dataset))


@tf.function
def tf_optimization_step():
    optimizer.minimize(training_loss, m.trainable_variables)


for epoch in range(epochs):
    for _ in range(num_batches_per_epoch):
        tf_optimization_step()
        # tf_optimization_step(model, training_loss, optimizer)
    epoch_id = epoch + 1
    if epoch_id % logging_epoch_freq == 0:
        tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")

gpf.utilities.print_summary(m)
mu, var = m.predict_y(Xnew)
fig, axs = plot_mean_and_var(xx, yy, mu.numpy(), var.numpy())
mu, var = m.predict_f(Xnew)
fig, axs = plot_mean_and_var(xx, yy, mu.numpy(), var.numpy())
plt.show()

lengthscales = m.kernel.lengthscales.numpy()
variance = m.kernel.variance.numpy()

q_mu = m.q_mu.numpy()
q_sqrt = m.q_sqrt.numpy()
z = m.inducing_variable.Z.numpy()
mean_func = m.mean_function.c.numpy()

np.savez(save_params_filename,
         l=lengthscales,
         var=variance,
         x=X,
         y=Y,
         mean_func=mean_func,
         q_mu=q_mu,
         q_sqrt=q_sqrt,
         z=z)
