# import matplotlib
import matplotlib.pyplot as plt
# import numpy as np
from jax import jacfwd
from jax import numpy as np
# import snips as snp
from scipy.integrate import ode
from scipy.optimize import least_squares, minimize

from derivative_kernel_gpy import DiffRBF
from geodesic_func import geodesic_fun

# snp.prettyplot(matplotlib)

# def compute_area_areaprime(x):
#     """
#     Compute the area and it's derivative as a function of independent variable x, return them as a vector.

#     :param x: independent variable, the domain of the problem is x=0 to L
#     :return: a 2-vec holding [A, dA/dx].
#     """
#     return [10, 0]  # Rectangle geometry


def compute_zprime(x, z, areafunction, X, Y, kernel):
    """
    Compute the value of the vector z's derivative at a point given the value of the vector function z and the
    independent variable x. The form of this calculation is specified by the vector ODE. Return a vector for the
    derivative.

    :param x: indpendent variable, the domain of the problem is x=0 to L
    :param z: 2-vec of the variable representing system of equations, z = [y, y']
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :return: 2-vec of values for z' = f(x, z)
    """
    # print(len(args))
    # areafunction, X, Y, kernel = args
    z_0 = np.array([z[0], z[1]])
    zprime_0 = np.array([z[2], z[3]])
    zprime_1 = geodesic_fun(z_0, zprime_0, X, Y, kernel)
    # zprime_1 = geodesic_fun(z_0, zprime_0)
    return [z[2], z[3], zprime_1[0, 0], zprime_1[1, 0]]


def integrate_over_domain(z_at_0,
                          integrator,
                          areafunction,
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
                 areafunction,
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
    integrator = ode(compute_zprime).set_integrator("dopri5")

    # z_at_0 = [y_at_0, 0.5]  # Make an initial guess for yprime at x=0

    def residuals(yprime_at_0, y_at_0, y_at_length):
        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        # print(yprime_at_0)
        print('Trying yprime_at_0: (%.8f, %.8f)' %
              (yprime_at_0[0], yprime_at_0[1]))
        # z_at_0 = y_at_0 + yprime_at_0
        z_at_0 = [y_at_0[0], y_at_0[1], yprime_at_0[0], yprime_at_0[1]]
        xs, zs = integrate_over_domain(z_at_0,
                                       integrator,
                                       areafunction,
                                       length=length,
                                       step=step,
                                       silent=silent)
        y_at_length_integrated = np.array(zs)[-1, 0:2]

        # Return the difference between y(L) found by numerical integrator and the true value
        error = y_at_length - y_at_length_integrated
        print('Residual [y(0)-y(L)] for current y\'(0): ', error)
        return error
        # return y_at_length - y_at_length_integrated

    lsq = least_squares(residuals,
                        yprime_at_0_guess,
                        args=(y_at_0, y_at_length),
                        loss="soft_l1")
    print('lsq')
    print(lsq)
    print(lsq.x)
    yprime_at_0_estimate = lsq.x[2]
    return yprime_at_0_estimate


setup_global_vars()

areafuncs = {"funnel1": lambda x: [20 / (x + 2) - 0.166, -20 / (x + 1)**2]}
fig, ax = plt.subplots()
integrator = ode(compute_zprime).set_integrator("dopri5")
y_at_length = [1, 2]
y_at_0 = [0, 0]
yprime_at_0 = [0.3, 1.2]
length = 3
step = 0.4
for nm, areaf in areafuncs.items():
    pjs = []
    z_at_0 = y_at_0 + yprime_at_0
    integrator.set_initial_value(z_at_0,
                                 t=0)  # Set the initial values of z and x
    integrator.set_f_params(areaf)
    print('before solve')
    yprime_at_0_estimate = solve_bvp_tj(y_at_0,
                                        y_at_length,
                                        yprime_at_0_guess=yprime_at_0,
                                        areafunction=areaf,
                                        length=length,
                                        step=step,
                                        silent=False)
    print('Optimised y\'(0): ', yprime_at_0_estimate)
    print(type(yprime_at_0_estimate))
    xs, zs = integrate_over_domain([y_at_0, yprime_at_0_estimate],
                                   integrator,
                                   areaf,
                                   length=length,
                                   step=step,
                                   silent=True)
    print('afer integrate')
    print(yprime_at_0_estimate)
    pjs.append(np.array(zs)[-1, 1])
    # ax.plot(tenvs, pjs, label=nm)
    ax.plot(xs, np.array(zs)[:, 0], label=nm)

ax.legend(loc="best")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5,
        0.95,
        "$T_j$ = %.0f" % (y_at_length, ),
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=props)
plt.tight_layout()

plt.show()

# areafuncs = {
#     "funnel1": lambda x: [20 / (x + 2) - 0.166, -20 / (x + 1)**2],
#     "funnel2": lambda x: [10 / (x + 1) - 0.59, -10 / (x + 1)**2],
#     "trapezoid": lambda x: [10 - 0.2 * x, -0.2],
#     "rectangle": lambda x: [10, 0]
# }
# areafunction = areafuncs['funnel1']
# # z_at_0 = [np.array([0, 0]), np.array([0.1, 0.1])]
# z_at_0 = [0, 0., 0.9, 0.9]
# # z_at_0 = [10., 0.1]
# xs, zs = integrate_over_domain(z_at_0, integrator, areafunction)
