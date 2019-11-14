import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parse_csv import parse_traj_csv, parse_vicon_csv

plot_start = False

# process data from vicon system
vicon_data, start_pos = parse_vicon_csv(file_name="5x5*.csv",
                                        folder_name="12-nov")
# time_vicon, x_vicon, y_vicon, z_vicon = parse_vicon_csv(file_name)
# time_vicon = np.arange(0, x_vicon.shape[0]) * 0.0093

# process data from tello
tello_data = parse_traj_csv('12-nov/traj.csv', time_interval=10)
print(start_pos)


def find_starting_index(vicon_data, x_cut=0.05, t_cut=20, plot_flag=False):
    """
    x_cut: set x velocity to mark beginning of trial
    t_cut: set time index to find max z at beginning
    """
    time_vicon = vicon_data['time']
    x_vicon = vicon_data['x'] / 10
    y_vicon = vicon_data['y'] / 10
    z_vicon = vicon_data['z'] / 10

    # calculate gradients for finding start of trial
    dx = np.zeros(x_vicon.shape[0])
    dy = np.zeros(x_vicon.shape[0])
    dz = np.zeros(x_vicon.shape[0])
    for i in range(1, x_vicon.shape[0]):
        dx[i] = x_vicon[i] - x_vicon[i - 1]
        dy[i] = y_vicon[i] - y_vicon[i - 1]
        dz[i] = z_vicon[i] - z_vicon[i - 1]

    high_idx = list(map(lambda i: i > t_cut, time_vicon)).index(True)

    # find index where drone reaches max speed (takeoff)
    z_idx = list(
        map(lambda i: i >= np.nanmax(dz[time_vicon < t_cut]),
            dz[time_vicon < t_cut])).index(True)

    # find when drone starts to move forward
    low_idx = z_idx
    low_idx = 0
    x_idx = list(map(lambda i: i > x_cut,
                     dx[low_idx:high_idx])).index(True) + low_idx

    # visualise starting point to make sure it is correct
    if plot_flag is True:
        # high_idx = list(map(lambda i: i > 4.2, time_vicon)).index(True)
        plt.plot(time_vicon[low_idx:high_idx],
                 dx[low_idx:high_idx],
                 label='dx')
        plt.plot(time_vicon[low_idx:high_idx],
                 dy[low_idx:high_idx],
                 label='dy')
        plt.plot(time_vicon[low_idx:high_idx],
                 dz[low_idx:high_idx],
                 label='dz')
        plt.scatter(time_vicon[x_idx], dx[x_idx])
        plt.scatter(time_vicon[z_idx], dz[z_idx])
        plt.legend()
        plt.show()
        input()
        plt.close()

    low_idx = x_idx

    high_idx = x_vicon.shape[0]
    time_vicon = time_vicon - time_vicon[low_idx]
    if plot_flag is True:
        plt.plot(time_vicon[low_idx:high_idx],
                 x_vicon[low_idx:high_idx],
                 label='x')
        plt.plot(time_vicon[low_idx:high_idx],
                 y_vicon[low_idx:high_idx],
                 label='y')
        plt.legend()
        plt.show()
        input()
        plt.close()
    return low_idx


def calc_tello_traj(tello_data, trial, plot_flag=False):
    x_tello = np.zeros(tello_data[trial]['x'].shape[0])
    y_tello = np.zeros(tello_data[trial]['y'].shape[0])
    z_tello = np.zeros(tello_data[trial]['z'].shape[0])
    time_tello = tello_data[trial]['time']

    for i in range(0, time_tello.shape[0] - 1):
        x_tello[i + 1] = x_tello[i] + tello_data[trial]['x'][i]
        y_tello[i + 1] = y_tello[i] + tello_data[trial]['y'][i]
        z_tello[i + 1] = z_tello[i] + tello_data[trial]['z'][i]
    pos = int(start_pos[trial])
    if pos == 1:
        x = x_tello
        y = y_tello
    elif pos == 2:
        x = -y_tello
        y = x_tello
    elif pos == 3:
        x = -y_tello
        y = x_tello
    elif pos == 4:
        x = -x_tello
        y = -y_tello

    if plot_flag is True:
        # plt.scatter(time_tello, x)
        # plt.scatter(time_tello, y)
        # plt.plot(time_tello, x)
        # plt.plot(time_tello, y)
        # plt.show()
        # input()
        # plt.close()
        plt.scatter(x, y)
        plt.plot(x, y)
        plt.show()
        input()
        plt.close()
    return x, y, z_tello


plot_vicon_traj = True
plot_find_start = False
plot_tello_traj = True
plot_tello_traj = False
trials = [0, 2, 3]
# start_pos = [1, 3, 4]
for trial in trials:
    time_vicon = vicon_data[trial]['time']
    start_idx = find_starting_index(vicon_data[trial],
                                    x_cut=0.05,
                                    t_cut=20,
                                    plot_flag=plot_find_start)

    time_vicon = time_vicon - time_vicon[start_idx]
    time_vicon = time_vicon[start_idx:]
    x_vicon = vicon_data[trial]['x'][start_idx:]
    y_vicon = vicon_data[trial]['y'][start_idx:]
    z_vicon = vicon_data[trial]['z'][start_idx:]

    x_tello, y_tello, z_tello = calc_tello_traj(tello_data,
                                                trial,
                                                plot_flag=plot_tello_traj)

    time_tello = tello_data[trial]['time']
    time_tello = time_tello - time_tello[0]
    dx_tello = tello_data[trial]['x']
    dy_tello = tello_data[trial]['y']
    dz_tello = tello_data[trial]['z']

    # get indices of time_vicon that correspond to those of time_tello
    idx = []
    for ii in range(0, time_tello.shape[0]):
        idx.append(
            list(map(lambda i: i >= time_tello[ii], time_vicon)).index(True))

    x_vicon_ = x_vicon[idx]
    y_vicon_ = y_vicon[idx]
    time_vicon_ = time_vicon[idx]
    dx = np.zeros(x_vicon_.shape[0])
    dy = np.zeros(y_vicon_.shape[0])
    for i in range(1, dx.shape[0] - 1):
        # dx_vicon[i] = x_vicon_[i] - x_tello[i] + dx_vicon[i - 1]
        # dx_vicon[i] = x_vicon_[i] - x_tello[i] + abs(dx_vicon[i - 1])
        dx[i] = x_vicon_[i] - x_vicon_[i - 1] - dx_tello[i] - dx[i - 1]
        dy[i] = y_vicon_[i] - y_vicon_[i - 1] - dy_tello[i] - dy[i - 1]

    # trial_input = np.stack([x_vicon_, y_vicon_]).T
    # x_tello = x_tello.r
    plt.plot(x_vicon, y_vicon, label=trial)
    plt.plot(x_tello + x_vicon[0], y_tello + y_vicon[0], label=trial)
    trial_input = np.stack([x_vicon_, y_vicon_]).T
    trial_output = np.stack([dx, dy]).T
    if trial == 0:
        model_input = trial_input
        model_output = trial_output
        time = time_vicon_
    else:
        model_input = np.concatenate([model_input, trial_input])
        model_output = np.concatenate([model_output, trial_output])
        time = np.concatenate([time, time_vicon_])
    # times.append(time_vicon_)

plt.legend()
input()
plt.close()

print(model_input.shape)
print(model_output.shape)
np.savez("uav_data1.npz", x=model_input, y=model_output)
# plt.scatter(time, model_output[:, 0], label='dx')
# plt.scatter(time, model_output[:, 1], label='dy')
# plt.legend()
# plt.show()
# input()
# plt.close()

# plt.scatter(model_input[:, 0], model_input[:, 1])
# plt.plot(model_input[:, 0], model_input[:, 1])
# plt.scatter(time, model_input[:, 0], label='x')
# plt.scatter(time, model_input[:, 1], label='y')
# plt.legend()

# delta_x = x_tello - x_vicon[idx]
# delta_y = y_tello - y_vicon[idx]
# print(delta_x)
# print(delta_x.shape)
# delta_x = delta_x[:-1].reshape(-1, 2)
# delta_y = delta_y[:-1].reshape(-1, 2)

# x = np.array([0, 500])
# y = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
# xx = np.tile(x, 11)
# yy = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
# yy = np.repeat(y, 2)

# fig = plt.figure(figsize=(12, 4))
# ax = fig.gca()
# surf = ax.contourf(x, y, delta_x, linewidth=0, antialiased=False)
# plt.scatter(xx, yy)
# plt.show()
# # input()

# fig = plt.figure(figsize=(12, 4))
# ax = fig.gca()
# surf = ax.contourf(np.array([0, 500]),
#                    y,
#                    delta_y,
#                    linewidth=0,
#                    antialiased=False)
# plt.scatter(xx, yy)
# plt.show()
# input()

# print(x_tello)
# # plt.scatter(x_tello, delta_x)
# # plt.scatter(y_tello, delta_y)
# plt.show()
# input()
