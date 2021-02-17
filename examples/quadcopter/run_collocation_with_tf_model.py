import gpjax
import matplotlib.pyplot as plt
from gpjax.kernels import RBF
from jax import numpy as np
from jax.config import config
from mogpe.training.utils import load_model_from_config_and_checkpoint

config.update("jax_enable_x64", True)

def init_svgp_gpjax_from_npz(filename):
    likelihood = None
    num_latent_gps = 1
    whiten = True
    mean_function = (
        gpjax.mean_functions.Zero()
    )  # mogpe gating functions have zero mean function
    # mean_func = params['mean_func']

    # Load svgp params and data
    params = np.load(filename)
    X = params["x"]  # [num_data x 2]
    Z = params["z"]  # [num_data x 2]

    # inducing points
    q_diag = False
    q_mu = params["q_mu"]  # [num_data x 1]
    q_sqrt = params["q_sqrt"]  # [num_data x 1]
    inducing_variable = (
        gating_func.inducing_variable.inducing_variable.Z.numpy()
    )

    # kerenl parameters
    # variance = gating_func.kernel.kernels[0].variance.numpy()
    # lengthscales = gating_func.kernel.kernels[0].lengthscales.numpy()
    lengthscales = params["l"]  # [2]
    print("lengthscales")
    print(lengthscales.shape)
    variance = params["var"]  # [1]
    kernel = RBF(variance=variance, lengthscales=lengthscales)

    svgp = SVGP(
        kernel,
        likelihood,
        inducing_variable,
        mean_function,
        num_latent_gps,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )

    return svgp



#########################
# Configure solver params
#########################
maxiter = 500  # max number of iterations
num_col_points = 10  # number of collocation points to use in solver
lb_defect = -0.05
ub_defect = 0.05
lb_defect = -0.01  # works with cov=40 jitter=1e-4
ub_defect = 0.01


# Initialise solver times
t_init = -1.0  # start time
t_end = 1.0  # end time
times = np.linspace(t_init, t_end, num_col_points)

# Initial guess of state vector at collocation points (guess straight line traj)
pos_init = np.array([2.7, 1.2])
pos_end_targ = np.array([-2.7, -0.5])
pos_init = np.array([2.7, 2.0])
pos_end_targ = np.array([-2.6, -1.5])
vel_init_guess = np.array([0.0000005, 0.0000003])  # initial guess of velocity
# vel_init_guess = np.array([0.0005, 0.0003])  # initial guess of velocity
state_guesses = init_straight_trajectory(
    pos_init, pos_end_targ, vel_init_guess, num_col_points=num_col_points
)


################################
# Configure metric tensor params
################################
covariance_weight = 20.0
covariance_weight = 1.0
jitter_ode = 1e-6
# jitter_ode = 1e-9
jitter_metric = 1e-4

######################################################
# Load a SVGP gating function from an mogpe checkpoint
######################################################
expert_num = 1
mogpe_dir = "../../../../mogpe/"
# ckpt_dir = mogpe_dir + "examples/logs/quadcopter/two_experts/09-08-144846"
# config_file = mogpe_dir + "examples/quadcopter/configs/config_2_experts.toml"
# data_file = mogpe_dir + "examples/quadcopter/data/quadcopter_data.npz"


ckpt_npz_file = (
    mogpe_dir + "models/saved_models/quadcopter/10-23-160809-param_dict.pickle"
)
svgp = init_svgp_gpjax_from_npz(ckpt_npz_file)
# svgp = init_svgp_gpjax_from_mogpe_ckpt(
#     config_file, ckpt_dir, data_file, expert_num=expert_num
# )


metric_tensor = SVGPMetricTensor(
    gp=svgp, covariance_weight=covariance_weight, jitter=jitter_metric
)
ode = GeodesicODE(metric_tensor=metric_tensor, jitter=jitter_ode)

collocation_solver = CollocationGeodesicSolver(
    ode=ode,
    covariance_weight=covariance_weight,
    maxiter=maxiter,
)


plot_svgp_and_start_end(svgp, traj_init=state_guesses)
plt.show()

t = time.time()
geodesic_traj = collocation_solver.solve_trajectory(
    state_guesses=state_guesses,
    pos_init=pos_init,
    pos_end_targ=pos_end_targ,
    times=times,
    lb_defect=lb_defect,
    ub_defect=ub_defect,
)
duration = time.time() - t
print("Optimisation duration: ", duration)

# save_img_dir = "./images"
plot_svgp_and_start_end(svgp, traj_init=state_guesses, traj_opts=geodesic_traj)
# plt.savefig(
#     save_img_dir + "/init-and-opt-trajs-on-svgp-new.pdf", transparent=True
# )
plt.show()
# traj_save_dir = "./saved_trajectories/opt_traj.npy"
# np.save(traj_save_dir, geodesic_traj)
