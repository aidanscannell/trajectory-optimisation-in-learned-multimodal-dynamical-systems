import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_csv import (parse_traj_csv, parse_traj_csv_by_empyt_line,
                       parse_vicon_csv)

plot_start = False

# process data from vicon system
vicon_data, start_pos = parse_vicon_csv(file_name="2-7x7*.csv",
                                        folder_name=".")

# process data from tello
# tello_data = parse_traj_csv('traj.csv', time_interval=9)
tello_data = parse_traj_csv_by_empyt_line('traj2.csv')
print("start_pos: " + str(start_pos))


def find_starting_index(vicon_data, t_cut=20, plot_flag=False):
    """
    x_cut: set x velocity to mark beginning of trial
    t_cut: set time index to find max z at beginning
    """
    time_vicon = vicon_data['time']
    x_vicon = vicon_data['x'] / 10
    y_vicon = vicon_data['y'] / 10
    z_vicon = vicon_data['z'] / 10

    high_idx = list(map(lambda i: i > t_cut, time_vicon)).index(True)

    def onpick(event):
        print('onpick idx:', event.ind)
        idx.append(event.ind)
        plt.close()
        fig.canvas.mpl_disconnect(cid)
        return True

    # visualise starting point to make sure it is correct
    idx = []
    low_idx = 0
    if plot_flag is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Select start of trial')
        ax.plot(time_vicon[low_idx:high_idx],
                x_vicon[low_idx:high_idx],
                label='x',
                picker=1)
        ax.plot(time_vicon[low_idx:high_idx],
                y_vicon[low_idx:high_idx],
                label='y',
                picker=1)
        plt.legend()
        cid = fig.canvas.mpl_connect('pick_event', onpick)

        plt.show(block=True)

    low_idx = idx[-1][0]
    print(low_idx)
    # low_idx = x_idx

    high_idx = x_vicon.shape[0]
    time_vicon = time_vicon - time_vicon[low_idx]
    if plot_flag is True:
        plt.figure()
        plt.title('Selected start point')
        plt.plot(time_vicon[:high_idx], x_vicon[:high_idx], label='x')
        plt.plot(time_vicon[:high_idx], y_vicon[:high_idx], label='y')
        plt.scatter(time_vicon[low_idx], x_vicon[low_idx])
        plt.scatter(time_vicon[low_idx], y_vicon[low_idx])
        plt.legend()
        plt.show(block=True)
    plt.close()
    # np.savez("start_idx.npz", low_idx)
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
        # plt.figure()
        # plt.scatter(time_tello, x)
        # plt.scatter(time_tello, y)
        # plt.plot(time_tello, x)
        # plt.plot(time_tello, y)
        # plt.show(block=True)
        fig = plt.figure()
        plt.scatter(x, y)
        plt.plot(x, y)
        plt.show(block=True)
    return x, y, z_tello


find_start_idx = True
find_start_idx = False

plot_vicon_traj = True
# plot_vicon_traj = False
plot_tello_traj = True
plot_tello_traj = False

trials = [0, 1, 2, 3]
if find_start_idx is False:
    start_idxs = np.load('start_idxs.npz')['x']
    plot_find_start = False
else:
    start_idxs = np.zeros(len(trials))
    plot_find_start = True

for trial in trials:
    time_vicon = vicon_data[trial]['time']
    if find_start_idx is True:
        start_idx = find_starting_index(vicon_data[trial],
                                        t_cut=30,
                                        plot_flag=plot_find_start)
        start_idxs[trial] = start_idx
    else:
        start_idx = int(start_idxs[trial])

    print('start idx: ', str(start_idx))
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
    if plot_vicon_traj is True:
        fig = plt.figure()
        # plt.plot(x_vicon, y_vicon, label=trial)
        # plt.plot(x_tello + x_vicon[0], y_tello + y_vicon[0], label=trial)

        # # plt.plot(time_vicon_, y_tello + y_vicon[0], label=trial)
        # # plt.plot(time_vicon_, x_tello + x_vicon[0], label=trial)
        # plt.scatter(x_tello + x_vicon[0], y_tello + y_vicon[0], label=trial)
        # plt.show(block=True)

        plt.plot(time_vicon, y_vicon, label=trial)
        plt.plot(time_vicon, x_vicon, label=trial)
        # plt.scatter(time_vicon_, y_vicon_, label=trial)
        # plt.scatter(time_vicon_, x_vicon_, label=trial)
        for t in time_vicon_:
            plt.axvline(t)
        # plt.plot(time_vicon_, y_tello + y_vicon[0], label=trial)
        # plt.plot(time_vicon_, x_tello + x_vicon[0], label=trial)
        # plt.scatter(time_vicon_, y_tello + y_vicon[0], label=trial)
        # plt.scatter(time_vicon_, x_tello + x_vicon[0], label=trial)
        plt.show(block=True)
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

if find_start_idx is True:
    np.savez("start_idxs.npz", x=start_idxs)

plt.figure()
plt.quiver(model_input[:, 0], model_input[:, 1], model_output[:, 0],
           model_output[:, 1])
plt.show(block=True)

print(model_input.shape)
print(model_output.shape)
np.savez("uav_data1.npz", x=model_input, y=model_output)
# plt.scatter(time, model_output[:, 0], label='dx')
# plt.scatter(time, model_output[:, 1], label='dy')
# plt.legend()
# plt.show()
# plt.close()

# plt.scatter(model_input[:, 0], model_input[:, 1])
# plt.plot(model_input[:, 0], model_input[:, 1])
# plt.scatter(time, model_input[:, 0], label='x')
# plt.scatter(time, model_input[:, 1], label='y')
# plt.legend()
