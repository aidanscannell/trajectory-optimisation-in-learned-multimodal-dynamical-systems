#!/usr/bin/env python3
from tromp.helpers import init_straight_trajectory
import jax.numpy as jnp

VEL_INIT_GUESS = jnp.array([0.0005, 0.0003])  # initial guess of velocity
VEL_INIT_GUESS = jnp.array([0.0000005, 0.0000003])
NUM_COL_POINTS = 10


class DiscreteTrajectory:
    def __init__(self, pos_init, pos_end_targ, state_guesses=None, times=None):
        self.pos_init = pos_init
        self.pos_end_targ = pos_end_targ
        self.pos_dim = pos_init.shape[0]
        self.state_dim = 2 * self.pos_dim

        if state_guesses is None and times is None:
            self.num_col_points = NUM_COL_POINTS
        elif state_guesses is None:
            self.num_col_points = times.shape[0]
        elif times is None:
            self.num_col_points = state_guesses.shape[0]

        if state_guesses is None:
            self.state_guesses = init_straight_trajectory(
                pos_init,
                pos_end_targ,
                vel_init_guess=VEL_INIT_GUESS,
                num_col_points=self.num_col_points,
            )
        else:
            self.state_guesses = state_guesses

        if times is None:
            t_init = -1.0  # start time
            # t_init = 0.0  # start time
            t_end = 1.0  # end time
            self.times = jnp.linspace(t_init, t_end, self.num_col_points)
        else:
            self.times = times

    def plot(self, fig, ax):
        if self.pos_dim == 2:
            ax.scatter(self.state_guesses[:, 0], self.state_guesses[:, 1])
            ax.plot(self.state_guesses[:, 0], self.state_guesses[:, 1])
        elif self.pos_dim == 2:
            ax.scatter(self.times, self.state_guesses)
            ax.plot(self.times, self.state_guesses)
        else:
            print("plotting is only supported in 1D and 2D")
        return fig, ax

    def state_guesses_to_opt_vars(self, state_guesses):
        """Remove start state from optimisation variables"""
        state_guesses = state_guesses[self.pos_dim :]
        state_guesses_before = state_guesses[: -self.state_dim]
        state_guesses_end = state_guesses[-self.state_dim + self.pos_dim :]
        return jnp.concatenate([state_guesses_before, state_guesses_end], axis=0)

    def opt_vars_to_state_guesses(self, opt_vars):
        # TODO replace referecne to num_states
        num_states = self.num_col_points
        state_guesses = opt_vars[: num_states * self.state_dim - 2 * self.pos_dim]
        # Add start pos
        state_guesses = jnp.concatenate([self.pos_init, state_guesses], axis=0)

        # Split state_guesses and insert end pos
        state_guesses_before = state_guesses[: -self.pos_dim]
        vel_end = state_guesses[-self.pos_dim :]
        state_guesses = jnp.concatenate(
            [state_guesses_before, self.pos_end_targ], axis=0
        )
        state_guesses = jnp.concatenate([state_guesses, vel_end], axis=0)

        state_guesses = state_guesses.reshape([num_states, self.state_dim])
        return state_guesses


# class HermiteTrajectory:
#     def __init__(self, pos_init, pos_end, state_guesses=None, times=None):
#         self.pos_init = pos_init
#         self.pos_end = pos_end

#         if state_guesses is None and times is None:
#             num_col_points = NUM_COL_POINTS
#         elif state_guesses is None:
#             num_col_points = times.shape[0]
#         elif times is None:
#             num_col_points = state_guesses.shape[0]

#         if state_guesses is None:
#             self.state_guesses = init_straight_trajectory(
#                 pos_init,
#                 pos_end,
#                 vel_init_guess=VEL_INIT_GUESS,
#                 num_col_points=num_col_points,
#             )
#         else:
#             self.state_guesses = state_guesses

#         if times is None:
#             t_init = -1.0  # start time
#             # t_init = 0.0  # start time
#             t_end = 1.0  # end time
#             self.times = jnp.linspace(t_init, t_end, num_col_points)
#         else:
#             self.times = times

#     def plot(self, fig, ax):
#         if self.pos_dim == 2:
#             ax.scatter(self.state_guesses[:, 0], self.state_guesses[:, 1])
#             ax.plot(self.state_guesses[:, 0], self.state_guesses[:, 1])
#         elif self.pos_dim == 2:
#             ax.scatter(self.times, self.state_guesses)
#             ax.plot(self.times, self.state_guesses)
#         else:
#             print("plotting is only supported in 1D and 2D")
#         return fig, ax


# def state_guesses_to_opt_vars(state_guesses):
#     # Flatten state guesses
#     state_dim = state_guesses.shape[1]
#     pos_dim = int(state_dim / 2)
#     num_states = state_guesses.shape[0]
#     state_guesses = state_guesses.reshape(-1)

#     # Remove start state from optimisation variables
#     state_guesses = state_guesses[pos_dim:]
#     print("state guesses removed start pos")
#     print(state_guesses.shape)
#     # Remove end pos from optimisation variables
#     # state_guesses = state_guesses[:-state_dim]
#     # end_state_start_idx = num_states * state_dim - state_dim - pos_dim
#     # end_pos_idxs = jnp.arange(end_state_start_idx,end_state_start_idx+pos_dim)
#     # print("end_pos_idxs")
#     # print(end_pos_idxs)
#     # state_guesses = jnp.delete(state_guesses, end_pos_idxs)
#     state_guesses_before = state_guesses[:-state_dim]
#     print("state_guesses_before")
#     print(state_guesses_before.shape)
#     state_guesses_end = state_guesses[-state_dim + pos_dim :]
#     print("state_guesses_end")
#     print(state_guesses_end.shape)
#     state_guesses = jnp.concatenate([state_guesses_before, state_guesses_end], axis=0)
#     print("state guesses removed end pos")
#     print(state_guesses.shape)
#     return state_guesses


# def opt_vars_to_states(opt_vars, pos_init, pos_end_targ, num_states):
#     if len(pos_init.shape) == 1:
#         pos_dim = pos_init.shape[0]
#         state_dim = 2 * pos_dim
#     state_guesses = opt_vars[: num_states * state_dim - 2 * pos_dim]
#     # Add start pos
#     state_guesses = jnp.concatenate([pos_init, state_guesses], axis=0)

#     # Split state_guesses and insert end pos
#     state_guesses_before = state_guesses[:-pos_dim]
#     vel_end = state_guesses[-pos_dim:]
#     state_guesses = jnp.concatenate([state_guesses_before, pos_end_targ], axis=0)
#     state_guesses = jnp.concatenate([state_guesses, vel_end], axis=0)

#     state_guesses = state_guesses.reshape([num_states, state_dim])
#     return state_guesses


# def opt_vars_to_states_and_lagrange(opt_vars, pos_init, pos_end_targ, num_states):
#     if len(pos_init.shape) == 1:
#         pos_dim = pos_init.shape[0]
#         state_dim = 2 * pos_dim
#     state_guesses = opt_vars_to_states(opt_vars, pos_init, pos_end_targ, num_states)

#     lagrange_multipliers = opt_vars[num_states * state_dim - 2 * pos_dim :]
#     return state_guesses, lagrange_multipliers
