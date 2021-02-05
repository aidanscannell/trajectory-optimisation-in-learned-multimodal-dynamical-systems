from tromp.plotting.trajectories import plot_trajs_over_svgp


def plot_solver_trajs_over_svgp(solver):
    svgp = solver.ode.metric_tensor.gp
    traj_init = solver.state_guesses
    traj_opt = solver.optimised_trajectory
    fig, axs = plot_trajs_over_svgp(svgp, traj_init=traj_init, traj_opt=traj_opt)
    return fig, axs
