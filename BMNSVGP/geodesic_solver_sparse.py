import datetime
import re
from pathlib import Path

import matplotlib.pyplot as plt
from jax import numpy as np
from jax import scipy as sp
from jax import vmap
from scipy.integrate import solve_ivp
from scipy.optimize import root
from derivative_kernel_gpy import DiffRBF

from probabilistic_geodesic import geodesic_fun
from utils.visualise_metric import (create_grid, gp_predict,
                                    load_data_and_init_kernel_fake,
                                    plot_gradient, plot_mean_and_var,
                                    plot_metric_trace, plot_mean_and_var)
from sparse_probabilistic_metric import (calc_G_map_sparse)


def compute_zprime(x, z, ode_func, ode_args):
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
    zprime_1 = ode_func(z_0, zprime_0, *ode_args)
    return [z[2], z[3], zprime_1[0, 0], zprime_1[1, 0]]


def integrate_over_domain(z_at_0,
                          ode_func,
                          ode_args,
                          integrator='RK45',
                          length=1,
                          step=0.2,
                          max_step=0.001,
                          silent=True):
    """
    Call runge-kutta repeatedly to integrate the vector function z over the full domain (specified by length). Return
    a list of 2-vecs which are the value of the vector z at every point in the domain discretized by 'step'. Note that
    runge-kutta object calls x as "t" and z as "y".

    :param z_at_0: the value of the vector z=[y, y'] at the left boundary point. should be list or array.
    :param integrator: a string for the integration method to use e.g. 'RK45', 'DOP853'
    :param length: the length of the domain to integrate on
    :param step: the step size with which to discretize the domain
    :param silent: bool indicating whether to print the progress of the integrator
    :return: array of 2-vecs - the value of vector z obtained by integration for each point in the discretized domain.
    """
    t = np.linspace(0., length, int(length / step))
    z_at_0 = np.array(z_at_0)

    integrator = solve_ivp(
        fun=lambda t, y: compute_zprime(t, y, ode_func, ode_args),
        t_span=(0., length),
        y0=z_at_0,
        t_eval=t,
        max_step=max_step,
        method=integrator)
    # dense_output=True)
    print(integrator)
    return integrator.t, integrator.y


def solve_bvp_tj(y_at_0,
                 y_at_length,
                 yprime_at_0_guess,
                 ode_func,
                 ode_args,
                 integrator,
                 length=1,
                 step=0.2,
                 root_tol=0.05,
                 maxfev=10000,
                 max_step=0.001,
                 silent=True):
    """
    Numerically find the value for y'(0) that gives us a propagated
    (integrated) solution matching most closely with other known boundary
    condition y(L) which is proportional junction temperature.

    :param y_at_0: the known boundary condition y(0)
    :param y_at_length: the known boundary condition y(L)
    :param yprime_at_0_guess: initial guess of y'(0)
    :param length: the length of the domain to integrate on
    :param step: the step size with which to discretize the domain
    :param silent: bool indicating whether to print the progress of the integrator
    :param root_tol: tolerance of root finder
    :return: the optimized estimate of y' at the left boundary point giving the most accurate integrated solution.
    """
    def residuals(yprime_at_0, y_at_0, y_at_length):
        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        print('Trying yprime_at_0: (%.8f, %.8f)' %
              (yprime_at_0[0], yprime_at_0[1]))
        z_at_0 = [y_at_0[0], y_at_0[1], yprime_at_0[0], yprime_at_0[1]]
        xs, zs = integrate_over_domain(z_at_0,
                                       ode_func,
                                       ode_args,
                                       integrator,
                                       length=length,
                                       step=step,
                                       max_step=max_step,
                                       silent=silent)
        # y_at_length_integrated = np.array(zs)[-1, 0:2]
        y_at_length_integrated = np.array(zs)[0:2, -1]

        # Return the difference between y(L) found by numerical integrator and the true value
        y_at_length = np.array(y_at_length)
        error = y_at_length - y_at_length_integrated
        print('Residual [y(0)-y(L)] for current y\'(0): ', error)
        return error

    yprime_at_0_guess = np.array(yprime_at_0_guess)

    rt = root(
        residuals,
        yprime_at_0_guess,
        args=(y_at_0, y_at_length),
        # jac=loss_jac,
        options={'maxfev': maxfev},
        tol=root_tol)
    yprime_at_0_estimate = rt.x
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
    h_mu = params['h_mu']  # [num_data x 1] mean of alpha
    h_var = params['h_var']  # [num_data x 1] variance of alpha
    m_h_mu = params['m_h_mu']  # [num_data x 1] mean of alpha
    m_h_var = params['m_h_var']  # [num_data x 1] variance of alpha
    xx = params['xx']
    yy = params['yy']
    xy = params['xy']
    mean_func = params['mean_func']

    kernel = DiffRBF(X.shape[1],
                     variance=var,
                     lengthscale=lengthscale,
                     ARD=True)
    return X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var


# Set parameters for integrator
length = 1.
step = 0.05
max_step = 0.001
integrator = 'RK45'
ode_func = geodesic_fun

# Set parameters for root finder
root_tol = 0.05
maxfev = 10000  # max function evaluations for root finder

# Set boundary conditions and create state vector at t=0
y_at_length = [-2., 2.5]
y_at_0 = [2., -2.8]
yprime_at_0 = [-5.64100643, 3.95933075]
z_at_0 = y_at_0 + yprime_at_0

# Load probabilistic manifold (GP) data
# X, a_mu, a_var, kernel = load_data_and_init_kernel_fake(
#     filename='saved_models/params_fake.npz')
# Y = a_mu
X, Y, h_mu, h_var, z, q_mu, q_sqrt, kernel, mean_func, xx, yy, xy, m_h_mu, m_h_var = load_data_and_init_kernel(
    filename='saved_models/27-2/137/params_from_model.npz')
Y = m_h_mu

# Initialise args for ode func (compute_zprime)
ode_args = (X, Y, kernel)

# Plot manifold with start and end points
xy, xx, yy = create_grid(X, N=961)
mu, cov = gp_predict(xy, X, Y, kernel)
var = np.diag(cov).reshape(-1, 1)
axs = plot_mean_and_var(xy, mu, var)

for ax in axs:
    # ax.plot(x_alpha, y_alpha, 'b')
    # ax.scatter(X[:, 0], X[:, 1], color='k', marker='x')
    ax.scatter(y_at_0[0], y_at_0[1], marker='o', color='r')
    ax.scatter(y_at_length[0], y_at_length[1], color='r', marker='o')
    ax.annotate("start", (y_at_0[0], y_at_0[1]))
    ax.annotate("end", (y_at_length[0], y_at_length[1]))

print('Calculating trace of metric, cov_j and mu_j...')
G, mu_j, cov_j = vmap(calc_G_map_sparse,
                      in_axes=(0, None, None, None, None, None, None,
                               None))(xy, X, z, q_mu, q_sqrt, kernel,
                                      mean_func, m_h_mu)
print('Done calculating metric')
# mu_j, var_j = gp_derivative_predict(xy, X, Y, kernel)
var_j = vmap(np.diag, in_axes=(0))(cov_j)
axs = plot_gradient(xy, mu_j, var_j, mu, var)
axs = plot_metric_trace(xy, G, mu, var)
plt.show()

# Solve the BVP using the shooting method (with a root finder)
yprime_at_0_estimate = solve_bvp_tj(y_at_0,
                                    y_at_length,
                                    yprime_at_0_guess=yprime_at_0,
                                    ode_func=ode_func,
                                    ode_args=ode_args,
                                    integrator=integrator,
                                    length=length,
                                    step=step,
                                    max_step=max_step,
                                    root_tol=root_tol,
                                    silent=False)

print('Optimised y\'(0): ', yprime_at_0_estimate)
z_at_0 = y_at_0 + list(yprime_at_0_estimate)
xs, zs = integrate_over_domain(z_at_0,
                               ode_func,
                               ode_args,
                               integrator,
                               length=length,
                               step=step,
                               max_step=max_step,
                               silent=False)

plt.close()
axs = plot_mean_and_var(xy, mu, var)
x = np.array(y_at_0).reshape(1, 2)
z = np.array(zs).T
z = np.append(x, z[:, 0:2], axis=0)
for ax in axs:
    ax.scatter(z[:, 0], z[:, 1], marker='x', color='k')
    ax.plot(z[:, 0], z[:, 1], marker='x', color='k')
    ax.scatter(y_at_0[0], y_at_0[1], marker='o', color='r')
    ax.scatter(y_at_length[0], y_at_length[1], color='r', marker='o')
    ax.annotate("start", (y_at_0[0], y_at_0[1]))
    ax.annotate("end", (y_at_length[0], y_at_length[1]))

date = datetime.datetime.now()
date_str = str(date.day) + "-" + str(date.month)
save_dirname = "../images/geodesic/" + date_str
Path(save_dirname).mkdir(parents=True, exist_ok=True)
traj_str = 'x1(0):' + str(y_at_0[0]) + '-x2(0):' + str(
    y_at_0[1]) + '--' + 'x1(L):' + str(y_at_length[0]) + '-x2(L):' + str(
        y_at_length[1])
traj_str = re.sub('[.]', '', traj_str)
plt.suptitle('Probabilistic Goedesic')
plt.savefig(save_dirname + '/' + traj_str + '.pdf', transparent=True)
plt.show()
