from functools import reduce

import matplotlib.pyplot as plt
from jax import jacfwd
from jax import numpy as np
from jax import scipy as sp
from jax.experimental import optimizers
from scipy.integrate import ode
from scipy.optimize import least_squares, minimize, root

from derivative_kernel_gpy import DiffRBF
# from geodesic_func import geodesic_fun
from geodesic_func_jac_and_val import geodesic_fun, geodesic_fun_sparse
from test_riemannian_metric import (create_grid, gp_predict, gp_predict_sparse,
                                    load_data_and_init_kernel,
                                    plot_mean_and_var)


def compute_zprime(x, z, X, mean_func, q_mu, q_sqrt, kernel):
    """
    Compute the value of the vector z's derivative at a point given the value of the vector function z and the
    independent variable x. The form of this calculation is specified by the vector ODE. Return a vector for the
    derivative.

    :param x: indpendent variable, the domain of the problem is x=0 to L
    :param z: 2-vec of the variable representing system of equations, z = [y, y']
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :return: 2-vec of values for z' = f(x, z)
    """
    z_0 = np.array([z[0], z[1]])
    zprime_0 = np.array([z[2], z[3]])
    # zprime_1 = geodesic_fun(z_0, zprime_0, X, Y, kernel)
    zprime_1 = geodesic_fun_sparse(z_0, zprime_0, X, mean_func, q_mu, q_sqrt,
                                   kernel)
    return [z[2], z[3], zprime_1[0, 0], zprime_1[1, 0]]


def integrate_over_domain(
        z_at_0,
        integrator,
        # areafunction,
        length=1,
        step=0.2,
        silent=True):
    """
    Call runge-kutta repeatedly to integrate the vector function z over the full domain (specified by length). Return
    a list of 2-vecs which are the value of the vector z at every point in the domain discretized by 'step'. Note that
    runge-kutta object calls x as "t" and z as "y".

    :param z_at_0: the value of the vector z=[y, y'] at the left boundary point. should be list or array.
    :param integrator: the runge-kutta numerical integrator object
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :param length: the length of the domain to integrate on
    :param step: the step size with which to discretize the domain
    :param silent: bool indicating whether to print the progress of the integrator
    :return: array of 2-vecs - the value of vector z obtained by integration for each point in the discretized domain.
    """
    initial_conditions = z_at_0
    integrator.set_initial_value(initial_conditions,
                                 t=0)  # Set the initial values of z and x
    # integrator.set_f_params(areafunction)
    dt = step

    xs, zs = [], []
    while integrator.successful() and integrator.t <= length:
        integrator.integrate(integrator.t + dt)
        xs.append(integrator.t)
        zs.append([
            integrator.y[0], integrator.y[1], integrator.y[2], integrator.y[3]
        ])
        if not silent:
            print("Current x and z values: ", integrator.t, integrator.y)
    return xs, zs


def solve_bvp_tj(y_at_0,
                 y_at_length,
                 yprime_at_0_guess,
                 integrator,
                 length=1,
                 step=0.2,
                 silent=True):
    """
    Numerically find the value for y'(0) that gives us a propagated (integrated) solution matching most closely with
    with other known boundary condition y(L) which is proportional junction temperature.

    :param y_at_0: the known boundary condition y(0)
    :param y_at_length: the known boundary condition y(L)
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :param length: the length of the domain to integrate on
    :param step: the step size with which to discretize the domain
    :param silent: bool indicating whether to print the progress of the integrator
    :return: the optimized estimate of y' at the left boundary point giving the most accurate integrated solution.
    """
    def loss(yprime_at_0, y_at_0, y_at_length):
        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        print('Trying yprime_at_0: (%.8f, %.8f)' %
              (yprime_at_0[0], yprime_at_0[1]))
        # z_at_0 = y_at_0 + yprime_at_0
        z_at_0 = [y_at_0[0], y_at_0[1], yprime_at_0[0], yprime_at_0[1]]
        xs, zs = integrate_over_domain(z_at_0,
                                       integrator,
                                       length=length,
                                       step=step,
                                       silent=silent)
        y_at_length_integrated = np.array(zs)[-1, 0:2]

        # Return the difference between y(L) found by numerical integrator and the true value
        y_at_length = np.array(y_at_length)
        error = y_at_length - y_at_length_integrated
        # print('Residual [y(0)-y(L)] for current y\'(0): ', error)
        loss = np.linalg.norm(error)
        print("Loss: ", loss)
        return loss

    def residuals(yprime_at_0, y_at_0, y_at_length):
        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        print('Trying yprime_at_0: (%.8f, %.8f)' %
              (yprime_at_0[0], yprime_at_0[1]))
        # z_at_0 = y_at_0 + yprime_at_0
        z_at_0 = [y_at_0[0], y_at_0[1], yprime_at_0[0], yprime_at_0[1]]
        xs, zs = integrate_over_domain(z_at_0,
                                       integrator,
                                       length=length,
                                       step=step,
                                       silent=silent)
        y_at_length_integrated = np.array(zs)[-1, 0:2]

        # Return the difference between y(L) found by numerical integrator and the true value
        y_at_length = np.array(y_at_length)
        error = y_at_length - y_at_length_integrated
        print('Residual [y(0)-y(L)] for current y\'(0): ', error)
        # return error[0]
        return y_at_length - y_at_length_integrated

    # loss_jac = jacfwd(residuals, 0)
    # print('jac')
    # print(loss_jac(yprime_at_0_guess, y_at_0, y_at_length))
    # print(len(loss_jac(yprime_at_0_guess, y_at_0, y_at_length)))

    yprime_at_0_guess = np.array(yprime_at_0_guess)

    rt = root(
        residuals,
        yprime_at_0_guess,
        args=(y_at_0, y_at_length),
        # jac=loss_jac,
        # tol=1.8,
        options={'disp': True})
    print('root')
    print(rt)
    yprime_at_0_estimate = rt.x

    # opt = minimize(
    #     loss,
    #     yprime_at_0_guess,
    #     args=(y_at_0, y_at_length),
    #     # jac=loss_jac,
    #     # tol=1.8,
    #     options={'disp': True})
    # print('opt')
    # print(opt)
    # yprime_at_0_estimate = opt.x[2]

    # lsq = least_squares(residuals,
    #                     yprime_at_0_guess,
    #                     args=(y_at_0, y_at_length),
    #                     loss="soft_l1",
    #                     verbose=2)
    # print('lsq')
    # print(lsq)
    # yprime_at_0_estimate = lsq.x
    # yprime_at_0_estimate = lsq.x[2]
    return yprime_at_0_estimate


def load_data_and_init_kernel(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscale = params['l']  # [2]
    var = params['var']  # [1]
    X = params['x']  # [num_data x 2]
    Y = params['y']  # [num_data x 2]
    z = params['z']  # [num_data x 2]
    q_mu = params['q_mu']  # [num_data x 1] mean of alpha
    q_sqrt = params['q_sqrt']  # [num_data x 1] variance of alpha
    a_mu = params['a_mu']  # [num_data x 1] mean of alpha
    a_var = params['a_var']  # [num_data x 1] variance of alpha
    mean_func = params['mean_func']

    kernel = DiffRBF(X.shape[1],
                     variance=var,
                     lengthscale=lengthscale,
                     ARD=True)
    return X, Y, a_mu, a_var, z, q_mu, q_sqrt, kernel, mean_func


x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 1000
y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 1000

# y_at_length = [1.5, 1.]
y_at_length = [-2., 2.5]
# y_at_0 = [-0.5, 0.]
# y_at_0 = [-0., -0.4]
y_at_0 = [2., -2.8]
# y_at_0 = [3., -2.]

# X, a_mu, a_var, kernel = load_data_and_init_kernel(
#     filename='saved_models/12-2/params_from_model.npz')
X, Y, a_mu, a_var, z, q_mu, q_sqrt, kernel, mean_func = load_data_and_init_kernel(
    filename='saved_models/15-2/params_from_model.npz')
# filename='saved_models/6-2/params_from_model.npz')
# X, a_mu, a_var, kernel = load_data_and_init_kernel(
#     filename='saved_models/params_fake.npz')


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + sp.special.erf(x / np.sqrt(2.0))) \
        * (1 - 2 * jitter) + jitter


xy, xx, yy = create_grid(X, N=961)

mu, var = gp_predict(xy, X, a_mu, kernel, mean_func=mean_func)
var = np.diag(var).reshape(-1, 1)
# axs = plot_mean_and_var(xy, mu, var)

mu, var = gp_predict_sparse(xy,
                            z,
                            mean_func,
                            q_mu,
                            q_sqrt,
                            kernel,
                            jitter=1e-4)
var = np.diag(var).reshape(-1, 1)
mu = inv_probit(mu / np.sqrt(1 + var))
var = mu - np.square(mu)
# axs_sparse = plot_mean_and_var(xy, mu, var)

# for axx in [axs, axs_sparse]:
#     for ax in axx:
#         ax.plot(x_alpha, y_alpha, 'b')
#         # ax.scatter(X[:, 0], X[:, 1], color='k', marker='x')
#         ax.scatter(y_at_0[0], y_at_0[1], marker='o', color='r')
#         ax.scatter(y_at_length[0], y_at_length[1], color='r', marker='o')
#         ax.annotate("start", (y_at_0[0], y_at_0[1]))
#         ax.annotate("end", (y_at_length[0], y_at_length[1]))
# plt.show()

fig, ax = plt.subplots()
# integrator = ode(compute_zprime).set_integrator("dopri5", nsteps=4000)
integrator = ode(compute_zprime).set_integrator("dopri5")
# integrator = ode(compute_zprime).set_integrator("vode")
# yprime_at_0 = [0.5, 1.2]

# yprime_at_0 = [1.36303914, 1.26927309]
# yprime_at_0 = [-3.63642215, 4.99931755]
yprime_at_0 = [-3.63642215, 1.99931755]
yprime_at_0_estimate = [-5.93384344, 2.26780623]

length = 1.
step = 0.05
pjs = []
z_at_0 = y_at_0 + yprime_at_0
integrator.set_initial_value(z_at_0, t=0)  # Set the initial values of z and x
integrator.set_f_params(z, mean_func, q_mu, q_sqrt, kernel)
# integrator.set_f_params(X, a_mu, kernel)
# integrator.set_f_params(areaf, X, a_mu, kernel)
print('before solve')
# yprime_at_0_estimate = solve_bvp_tj(y_at_0,
#                                     y_at_length,
#                                     yprime_at_0_guess=yprime_at_0,
#                                     integrator=integrator,
#                                     length=length,
#                                     step=step,
#                                     silent=False)
# yprime_at_0_estimate = [-3.99900522, 5.50012926]

print('Optimised y\'(0): ', yprime_at_0_estimate)
z_at_0 = y_at_0 + list(yprime_at_0_estimate)
xs, zs = integrate_over_domain(z_at_0,
                               integrator,
                               length=length,
                               step=step,
                               silent=False)

plt.close()
axs = plot_mean_and_var(xy, mu, var)
x = np.array(y_at_0).reshape(1, 2)
z = np.array(zs)
z = np.append(x, z[:, 0:2], axis=0)
for ax in axs:
    ax.scatter(z[:, 0], z[:, 1], marker='x', color='k')
    ax.plot(z[:, 0], z[:, 1], marker='x', color='k')
    ax.scatter(y_at_0[0], y_at_0[1], marker='o', color='r')
    ax.scatter(y_at_length[0], y_at_length[1], color='r', marker='o')
    ax.annotate("start", (y_at_0[0], y_at_0[1]))
    ax.annotate("end", (y_at_length[0], y_at_length[1]))
plt.savefig("../images/geodesic_fake.pdf", transparent=True)
plt.show()
