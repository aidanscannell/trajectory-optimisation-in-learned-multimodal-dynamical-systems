import pickle

from jax import numpy as np

from .kernels import DiffRBF


def load_data_and_init_kernel_fake(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscale = params["l"]  # [2]
    var = params["var"]  # [1]
    X = params["x"]  # [num_data x 2]
    Y = params["a_mu"]  # [num_data x 1] mean of alpha
    # a_var = params['a_var']  # [num_data x 1] variance of alpha
    kernel = DiffRBF(
        X.shape[1], variance=var, lengthscale=lengthscale, ARD=True
    )
    return X, Y, kernel


def load_data_and_init_kernel_sparse(filename):
    # Load kernel hyper-params and create kernel
    params = np.load(filename)
    lengthscale = params["l"]  # [2]
    var = params["var"]  # [1]
    X = params["x"]  # [num_data x 2]
    Z = params["z"]  # [num_data x 2]
    q_mu = params["q_mu"]  # [num_data x 1] mean of alpha
    q_sqrt = params["q_sqrt"]  # [num_data x 1] variance of alpha
    mean_func = params["mean_func"]
    kernel = DiffRBF(
        X.shape[1], variance=var, lengthscale=lengthscale, ARD=True
    )
    return X, Z, q_mu, q_sqrt, kernel, mean_func


# def load_data_and_init_kernel_mogpe(
#     save_model_dir="../../models/saved_models/quadcopter/09-08-144846-param_dict.pickle"
# ):

#     f = open(save_model_dir, "rb")
#     param_dict = pickle.load(f)
#     print(param_dict.keys())
#     gating_function_num = '0'
#     variance = param_dict['.gating_network.gating_function_list[' +
#                           gating_function_num +
#                           '].kernel.kernels[0].variance'].numpy()
#     lengthscale = param_dict['.gating_network.gating_function_list[' +
#                              gating_function_num +
#                              '].kernel.kernels[0].lengthscales'].numpy()  # [2]
#     q_mu = param_dict['.gating_network.gating_function_list[' +
#                       gating_function_num + '].q_mu'].numpy()
#     q_sqrt = param_dict['.gating_network.gating_function_list[' +
#                         gating_function_num + '].q_sqrt'].numpy()
#     Z = param_dict['.gating_network.gating_function_list[' +
#                    gating_function_num +
#                    '].inducing_variable.inducing_variable.Z'].numpy()
#     # mean_func = param_dict[
#     #     '.gating_network.gating_function_list[0].mean_function.c'].numpy()
#     mean_func = 0.

#     kernel = DiffRBF(Z.shape[1],
#                      variance=variance,
#                      lengthscale=lengthscale,
#                      ARD=True)
#     return Z, q_mu, q_sqrt, kernel, mean_func


def load_data_and_init_kernel_mogpe(
    expert_num="0",
    save_model_dir="../../models/saved_models/quadcopter/09-08-144846-param_dict.pickle",
):

    f = open(save_model_dir, "rb")
    param_dict = pickle.load(f)
    print(param_dict.keys())
    q_mu = param_dict[".gating_network.gating_function_list[0].q_mu"].numpy()
    output_dim = q_mu.shape[1]
    print("output_dim")
    print(output_dim)
    print(q_mu.shape)

    q_mu = param_dict[
        ".gating_network.gating_function_list[" + expert_num + "].q_mu"
    ].numpy()
    q_sqrt = param_dict[
        ".gating_network.gating_function_list[" + expert_num + "].q_sqrt"
    ].numpy()
    Z = param_dict[
        ".gating_network.gating_function_list["
        + expert_num
        + "].inducing_variable.inducing_variable.Z"
    ].numpy()

    noise_vars = []
    k = 0
    while True:
        try:
            noise_var = param_dict[
                ".experts.experts_list[" + str(k) + "].likelihood.variance"
            ].numpy()
            print("noise var = ", noise_var)
            noise_vars.append(noise_var)
            k += 1
        except:
            break

    kernels, mean_funcs = [], []
    for i in range(output_dim):
        variance = param_dict[
            ".gating_network.gating_function_list["
            + expert_num
            + "].kernel.kernels["
            + str(i)
            + "].variance"
        ].numpy()
        print("var ", i)
        print(variance)
        lengthscale = param_dict[
            ".gating_network.gating_function_list["
            + expert_num
            + "].kernel.kernels["
            + str(i)
            + "].lengthscales"
        ].numpy()  # [2]
        print("l")
        print(lengthscale)
        # mean_func = param_dict['.gating_network.gating_function_list[' +
        #                        expert_num + '].mean_function.c'].numpy()
        mean_func = 0.0

        kernel = DiffRBF(
            Z.shape[1], variance=variance, lengthscale=lengthscale, ARD=True
        )
        mean_funcs.append(mean_func)
        kernels.append(kernel)
    return Z, q_mu, q_sqrt, kernels, mean_funcs, noise_vars


# model = parse_model_from_config_file(config_file)

# gpf.utilities.print_summary(model)
# gpf.utilities.multiple_assign(model, param_dict)
# gpf.utilities.print_summary(model)
# return model

if __name__ == "__main__":
    load_data_and_init_kernel_mogpe()
