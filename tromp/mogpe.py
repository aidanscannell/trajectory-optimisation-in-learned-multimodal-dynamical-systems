import jax
from gpjax.custom_types import InputData, MeanAndVariance, MeanFunc, OutputData
from gpjax.kernels import Kernel
# from ProbGeo.gp.gp import (InputData, MeanAndVariance, MeanFunc, OutputData,
#                            gp_predict)
from gpjax.prediction import gp_predict
from jax import jacfwd, jacrev
from jax import numpy as np


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return (
        0.5 * (1.0 + jax.lax.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter
    )


def bernoulli_predict_mean_and_var(Fmu, Fvar):
    prob = inv_probit(Fmu / np.sqrt(1 + Fvar))
    return prob, prob - np.square(prob)


def single_mogpe_mixing_probability(
    Xnew: InputData,
    X: InputData,
    kernel: Kernel,
    mean_func: MeanFunc,
    f: OutputData,
    full_cov: bool = False,
    q_sqrt=None,
    jitter=1e-6,
    white: bool = True,
):
    """
    TODO are these dimensions the right way around?
    returns: [num_test, output_dim, input_dim]
    """
    if len(Xnew.shape) == 1:
        Xnew = Xnew.reshape(1, -1)
    mu, var = gp_predict(
        Xnew,
        X,
        kernel,
        mean_func,
        f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        jitter=jitter,
        white=white,
    )
    print("inside single mixing prob")
    print(mu.shape)
    print(var.shape)
    output_dim = f.shape[-1]
    if full_cov is False:
        var = var.reshape(-1, output_dim)
    prob_a_0s = []
    for i in range(output_dim):
        prob_a_0, _ = bernoulli_predict_mean_and_var(
            mu[i : i + 1, :], var[:, i : i + 1]
        )
        prob_a_0s.append(prob_a_0.reshape(1))
    prob_a_0 = np.stack(prob_a_0s)  # [input_dim , 1]
    print("prob a 0")
    print(prob_a_0.shape)

    # product over output dimensions
    prob_a_0 = np.prod(prob_a_0)
    print(prob_a_0.shape)
    # return prob_a_0
    return prob_a_0.reshape(-1)
    # return 10 * prob_a_0.reshape(-1)


def mogpe_mixing_probability(
    Xnew: InputData,
    X: InputData,
    kernel: Kernel,
    mean_func: MeanFunc,
    f: OutputData,
    full_cov: bool = False,
    q_sqrt=None,
    jitter=1e-6,
    white: bool = True,
):
    """
    TODO are these dimensions the right way around?
    returns: [num_test, output_dim, input_dim]
    """
    if len(Xnew.shape) == 1:
        Xnew = Xnew.reshape(1, -1)
    mu, var = gp_predict(
        Xnew,
        X,
        kernel,
        mean_func,
        f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        jitter=jitter,
        white=white,
    )
    output_dim = f.shape[-1]
    if full_cov is False:
        var = var.reshape(-1, output_dim)
    prob_a_0, _ = bernoulli_predict_mean_and_var(mu, var)
    print("prob_a_0")
    print(prob_a_0.shape)
    return 10 * prob_a_0
    # prob_a_1 = 1 - prob_a_0
    # mixing_probs = np.stack([prob_a_0, prob_a_1], -1)
    # return mixing_probs


# def single_mogpe_mixing_probability(Xnew: InputData,
#                                     X: InputData,
#                                     kernel: Kernel,
#                                     mean_func: MeanFunc,
#                                     f: OutputData,
#                                     full_cov: bool = False,
#                                     q_sqrt=None,
#                                     jitter=1e-6,
#                                     white: bool = True):
#     """
#     TODO are these dimensions the right way around?
#     returns: [num_test, output_dim, input_dim]
#     """
#     if len(Xnew.shape) == 1:
#         Xnew = Xnew.reshape(1, -1)
#     mu, var = gp_predict(Xnew,
#                          X,
#                          kernel,
#                          mean_func,
#                          f,
#                          full_cov=full_cov,
#                          q_sqrt=q_sqrt,
#                          jitter=jitter,
#                          white=white)
#     # return mu.reshape(-1)
#     output_dim = f.shape[-1]
#     if full_cov is False:
#         var = var.reshape(-1, output_dim)
#     prob_a_0, _ = bernoulli_predict_mean_and_var(mu, var)
#     print('prob_a_0')
#     print(prob_a_0.shape)
#     return prob_a_0.reshape(-1)
#     # return 10 * prob_a_0.reshape(-1)

# def mogpe_mixing_probability(Xnew: InputData,
#                              X: InputData,
#                              kernel: Kernel,
#                              mean_func: MeanFunc,
#                              f: OutputData,
#                              full_cov: bool = False,
#                              q_sqrt=None,
#                              jitter=1e-6,
#                              white: bool = True):
#     """
#     TODO are these dimensions the right way around?
#     returns: [num_test, output_dim, input_dim]
#     """
#     if len(Xnew.shape) == 1:
#         Xnew = Xnew.reshape(1, -1)
#     mu, var = gp_predict(Xnew,
#                          X,
#                          kernel,
#                          mean_func,
#                          f,
#                          full_cov=full_cov,
#                          q_sqrt=q_sqrt,
#                          jitter=jitter,
#                          white=white)
#     output_dim = f.shape[-1]
#     if full_cov is False:
#         var = var.reshape(-1, output_dim)
#     prob_a_0, _ = bernoulli_predict_mean_and_var(mu, var)
#     print('prob_a_0')
#     print(prob_a_0.shape)
#     return 10 * prob_a_0
#     # prob_a_1 = 1 - prob_a_0
#     # mixing_probs = np.stack([prob_a_0, prob_a_1], -1)
#     # return mixing_probs


def mogpe_mixing_prob_jacobian(
    Xnew: InputData,
    X: InputData,
    kernel: Kernel,
    mean_func: MeanFunc,
    f: OutputData,
    full_cov: bool = False,
    q_sqrt=None,
    jitter=1e-6,
    white: bool = True,
) -> MeanAndVariance:
    """
    TODO are these dimensions the right way around?
    returns: [num_test, num_experts, input_dim]
    """

    def single_jac(x_new):
        x_new = x_new.reshape(1, -1)
        jac = jacfwd(mogpe_mixing_probability, (0))(
            x_new, X, kernel, mean_func, f, full_cov, q_sqrt, jitter, white
        )
        return jac

    input_dim = Xnew.shape[1]
    output_dim = f.shape[1]
    num_test = Xnew.shape[0]
    jac = jax.vmap(single_jac)(Xnew)
    print("og jac")
    print(jac.shape)
    # TODO is output dim handled correctly here?
    jac = jac.reshape(num_test, output_dim, input_dim, -1)

    print("jac")
    print(jac.shape)
    return jac
