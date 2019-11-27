import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_csv import (parse_traj_csv, parse_traj_csv_by_empyt_line,
                       parse_vicon_csv)

# process data from vicon system
vicon_data, start_pos = parse_vicon_csv(file_name="26nov-7x7*.csv",
                                        folder_name=".")

# process data from tello
# tello_data = parse_traj_csv('traj.csv', time_interval=9)
tello_data = parse_traj_csv_by_empyt_line('traj2.csv')
print("start_pos: " + str(start_pos))

# position of sheet
x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 10
y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 10


def calc_tello_traj(tello_data, trial):

    x_tello = np.zeros(tello_data[trial]['x'].shape[0])
    y_tello = np.zeros(tello_data[trial]['y'].shape[0])
    z_tello = np.zeros(tello_data[trial]['z'].shape[0])
    time_tello = tello_data[trial]['time']

    for i in range(0, time_tello.shape[0] - 1):
        x_tello[i + 1] = x_tello[i] + tello_data[trial]['x'][i]
        y_tello[i + 1] = y_tello[i] + tello_data[trial]['y'][i]
        z_tello[i + 1] = z_tello[i] + tello_data[trial]['z'][i]

    dx_tello = tello_data[trial]['x']
    dy_tello = tello_data[trial]['y']

    pos = int(start_pos[trial])
    if pos == 1:
        x = x_tello
        y = y_tello
        dx = dx_tello
        dy = dy_tello
    elif pos == 2:
        x = y_tello
        y = -x_tello
        dx = dy_tello
        dy = -dx_tello
    elif pos == 3:
        x = -y_tello
        y = x_tello
        dx = -dy_tello
        dy = dx_tello
    elif pos == 4:
        x = -x_tello
        y = -y_tello
        dx = -dx_tello
        dy = -dy_tello

    # x = x / np.cos(rz_vicon)
    # y = y / np.cos(rz_vicon)
    # dx = dx / np.cos(rz_vicon)
    # dy = dy / np.cos(rz_vicon)

    return x, y, dx, dy


def onpick(event):
    print('onpick idx:', event.ind)
    global idx
    idx = event.ind
    plt.close()
    # fig.canvas.mpl_disconnect(cid)
    return True


def find_starting_index(vicon_data, t_cut=20):
    """
    t_cut: set time index to find max z at beginning
    """
    time_vicon = vicon_data['time']
    x_vicon = vicon_data['x'] / 10
    y_vicon = vicon_data['y'] / 10
    z_vicon = vicon_data['z'] / 10

    high_idx = list(map(lambda i: i > t_cut, time_vicon)).index(True)

    # visualise starting point to make sure it is correct
    # idx = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select start of trial')
    ax.plot(time_vicon[:high_idx], x_vicon[:high_idx], label='x', picker=1)
    ax.plot(time_vicon[:high_idx], y_vicon[:high_idx], label='y', picker=1)
    plt.legend()
    cid = fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block=True)
    print('idx')
    print(idx)

    low_idx = idx[-1]
    return low_idx


def plot_vicon_with_quiver(vicon_data, tello_data, trial, start_idx):
    time_vicon = vicon_data[trial]['time']
    time_vicon = time_vicon - time_vicon[start_idx]
    time_vicon = time_vicon[start_idx:]
    x_vicon = vicon_data[trial]['x'][start_idx:]
    y_vicon = vicon_data[trial]['y'][start_idx:]

    time_tello = tello_data[trial]['time']
    time_tello = time_tello - time_tello[0]

    idx = get_vicon_idx_at_test_points(time_vicon, time_tello)

    x_vicon_at_test_points = x_vicon[idx]
    y_vicon_at_test_points = y_vicon[idx]
    time_vicon_at_test_points = time_vicon[idx]

    x_tello, y_tello, dx_tello, dy_tello = calc_tello_traj(tello_data, trial)

    plt.plot(x_vicon, y_vicon, label=trial)
    plt.plot(x_tello + x_vicon[0], y_tello + y_vicon[0], label=trial)
    plt.scatter(x_vicon_at_test_points,
                y_vicon_at_test_points,
                marker='x',
                c='k',
                zorder=10)
    plt.quiver(x_vicon_at_test_points,
               y_vicon_at_test_points,
               dx_tello,
               dy_tello,
               angles='xy',
               scale_units='xy',
               width=0.001,
               scale=1)
    for i in range(x_vicon_at_test_points.shape[0]):
        plt.annotate(i, (x_vicon_at_test_points[i], y_vicon_at_test_points[i]))

    plt.plot(x_alpha, y_alpha)


def calibrate_idxs(vicon_data, tell_data, trial, t_cut=20):

    start_idx = find_starting_index(vicon_data[trial], t_cut=t_cut)

    plt.figure()
    plot_vicon_with_quiver(vicon_data, tello_data, trial, start_idx)

    # plt.show(block=True)
    plt.show()
    return start_idx


def get_vicon_idx_at_test_points(time_vicon, time_tello):
    """ get indices of time_vicon that correspond to those of time_tello """
    idx = []
    for ii in range(0, time_tello.shape[0]):
        idx.append(
            list(map(lambda i: i >= time_tello[ii], time_vicon)).index(True))
    return idx


def calc_error(vicon_data, tello_data, trial, start_idxs):
    """ Calculate dx and dy at each test point """
    time_vicon = vicon_data[trial]['time']
    start_idx = int(start_idxs[trial])

    print('start idx: ', str(start_idx))
    time_vicon = time_vicon - time_vicon[start_idx]
    time_vicon = time_vicon[start_idx:]
    x_vicon = vicon_data[trial]['x'][start_idx:]
    y_vicon = vicon_data[trial]['y'][start_idx:]

    x_tello, y_tello, dx_tello, dy_tello = calc_tello_traj(tello_data, trial)
    time_tello = tello_data[trial]['time']
    time_tello = time_tello - time_tello[0]

    idx = get_vicon_idx_at_test_points(time_vicon, time_tello)

    x_vicon_at_test_points = x_vicon[idx]
    y_vicon_at_test_points = y_vicon[idx]
    time_vicon_at_test_points = time_vicon[idx]
    dx = np.zeros(len(idx))
    dy = np.zeros(len(idx))
    for i in range(1, dx.shape[0] - 1):
        dx[i] = x_vicon_at_test_points[i] - x_vicon_at_test_points[
            i - 1] - dx_tello[i - 1]
        dy[i] = y_vicon_at_test_points[i] - y_vicon_at_test_points[
            i - 1] - dy_tello[i - 1]
    return dx, dy, x_vicon_at_test_points, y_vicon_at_test_points


def create_data_for_model(vicon_data, tello_data, trials, start_idxs):
    for trial in trials:
        dx, dy, x_vicon_at_test_points, y_vicon_at_test_points = calc_error(
            vicon_data, tello_data, trial, start_idxs)
        # concatenate inputs and outputs from trial
        trial_input = np.stack(
            [x_vicon_at_test_points, y_vicon_at_test_points]).T
        trial_output = np.stack([dx, dy]).T
        if trial == 0:
            model_input = trial_input
            model_output = trial_output
            # time = time_vicon_at_test_points
        else:
            model_input = np.concatenate([model_input, trial_input])
            model_output = np.concatenate([model_output, trial_output])
            # time = np.concatenate([time, time_vicon_at_test_points])
    return model_input, model_output


def align_tello_data(tello_data, vicon_data, trials, start_idxs):

    for trial in trials:
        start_idx = int(start_idxs[trial])
        time_vicon = vicon_data[trial]['time']
        time_vicon = time_vicon - time_vicon[start_idx]
        time_vicon = time_vicon[start_idx:]
        x_vicon = vicon_data[trial]['x']
        y_vicon = vicon_data[trial]['y']

        time_tello = tello_data[trial]['time']
        time_tello = time_tello - time_tello[0]

        idx = get_vicon_idx_at_test_points(time_vicon, time_tello)
        rz_vicon = vicon_data[trial]['rz'][idx]

        dx_tello = tello_data[trial]['x']
        dy_tello = tello_data[trial]['y']
        dz_tello = tello_data[trial]['z']
        tello_matrix = np.stack([dx_tello, dy_tello, dz_tello], axis=1)
        tello_matrix = tello_matrix.reshape(-1, 1, 3)
        R = np.zeros([rz_vicon.shape[0], 3, 3])
        for i, (c, s) in enumerate(zip(np.cos(rz_vicon), np.sin(rz_vicon))):
            R[i, ...] = np.array([[-c, -s, 0], [s, -c, 0], [0, 0, 1]])
        rotated_tello = tello_matrix @ R
        rotated_tello = np.squeeze(rotated_tello.reshape(-1, 3))

        x_tello = np.zeros([rotated_tello[:, 0].shape[0], 1])
        y_tello = np.zeros([rotated_tello[:, 1].shape[0], 1])
        z_tello = np.zeros([rotated_tello[:, 2].shape[0], 1])
        for i in range(0, x_tello.shape[0] - 1):
            x_tello[i + 1, 0] = x_tello[i, 0] + rotated_tello[i, 0]
            y_tello[i + 1, 0] = y_tello[i, 0] + rotated_tello[i, 1]
            z_tello[i + 1, 0] = z_tello[i, 0] + rotated_tello[i, 2]

        dx_tello = rotated_tello[:, 0]
        dy_tello = rotated_tello[:, 1]

        plt.figure()
        # plt.plot(dx_tello_aligned, dy_tello_aligned)
        # plt.plot(dx, dy)
        # plt.plot(x, y, label='aligned tello')

        plt.plot(x_vicon, y_vicon, label='raw vicon')
        # plt.plot(x + x_vicon[0], y + y_vicon[0], label='aligned tello')
        plt.plot(x_tello + x_vicon[0],
                 y_tello + y_vicon[0],
                 label='aligned tello')
        # plt.scatter(x_vicon_at_test_points,
        #             y_vicon_at_test_points,
        #             marker='x',
        #             c='k',
        #             zorder=10)
        x_vicon_at_test_points = x_vicon[start_idx:][idx]
        y_vicon_at_test_points = y_vicon[start_idx:][idx]
        plt.quiver(x_vicon_at_test_points,
                   y_vicon_at_test_points,
                   dx_tello,
                   dy_tello,
                   angles='xy',
                   scale_units='xy',
                   width=0.001,
                   scale=1)
        # for i in range(x_vicon_at_test_points.shape[0]):
        #     plt.annotate(i, (x_vicon_at_test_points[i], y_vicon_at_test_points[i]))
        # plt.plot(x_alpha, y_alpha)
        plt.legend()
        plt.show(block=True)


trials = [0, 1, 2, 3]
start_idxs = np.load('start_idxs.npz')['x']
align_tello_data(tello_data, vicon_data, trials, start_idxs)

find_start_idx = True
find_start_idx = False

plot_quiver = True
plot_quiver = False
plot_data = True
plot_data = False
plot_checker = True
plot_checker = False

trials = [0, 1, 2, 3]

if find_start_idx is False:
    start_idxs = np.load('start_idxs.npz')['x']
else:
    start_idxs = np.zeros(len(trials))
    # find start index for each trial
    print('inside else')
    for trial in trials:
        print('inside trial')
        done = False
        while done is False:
            start_idx = calibrate_idxs(vicon_data, tello_data, trial, t_cut=20)
            print('before input')
            dt = input()
            print('after input')
            if dt is 'y':
                print('inside dt=0')
                done = True
        start_idxs[trial] = start_idx
    np.savez("start_idxs.npz", x=start_idxs)

if plot_quiver is True:
    for trial in trials:
        plot_vicon_with_quiver(vicon_data, tello_data, trial,
                               int(start_idxs[trial]))
        plt.show(block=True)

model_input, model_output = create_data_for_model(vicon_data, tello_data,
                                                  trials, start_idxs)
print(model_input.shape)
print(model_output.shape)

if plot_data is True:
    # visualise data using quiver plot
    plt.figure()
    plt.quiver(model_input[:, 0], model_input[:, 1], model_output[:, 0],
               model_output[:, 1])
    plt.show(block=True)

np.savez("uav_data1.npz", x=model_input, y=model_output)

if plot_checker is True:
    for trial in trials:
        dx, dy, x_vicon_at_test_points, y_vicon_at_test_points = calc_error(
            vicon_data, tello_data, trial, start_idxs)

        x_tello, y_tello, dx_tello, dy_tello = calc_tello_traj(
            tello_data, trial)
        fig = plt.figure()
        plot_vicon_with_quiver(vicon_data, tello_data, trial,
                               int(start_idxs[trial]))
        # print(x_vicon_at_test_points)
        # print(dx_tello.shape)
        # print(dx.shape)
        # print(dy.shape)
        plt.quiver(x_vicon_at_test_points[:-1] + dx_tello[:-1],
                   y_vicon_at_test_points[:-1] + dy_tello[:-1],
                   dx[1:],
                   dy[1:],
                   angles='xy',
                   scale_units='xy',
                   width=0.001,
                   color='r',
                   scale=1)
        plt.show(block=True)

        # fig = plt.figure()
        # plt.plot(time_vicon, y_vicon, label=trial)
        # plt.plot(time_vicon, x_vicon, label=trial)
        # plt.scatter(time_vicon_at_test_points,
        #             x_vicon_at_test_points,
        #             zorder=10,
        #             c='k',
        #             marker='|')
        # plt.scatter(time_vicon_at_test_points,
        #             y_vicon_at_test_points,
        #             zorder=10,
        #             c='k',
        #             marker='|')
        # # for t in time_vicon_at_test_points:
        # #     plt.axvline(t)
        # plt.show(block=True)

# done = False
# dt = 0
# while not done:
#     fig = plt.figure()
#     plt.title("Start pos " + str(start_pos[trial]))
#     plt.plot(time_vicon, y_vicon, label='x')
#     plt.plot(time_vicon, x_vicon, label='y')
#     for t in time_vicon_at_tello_idx:
#         plt.axvline(t + dt)
#     plt.legend()
#     # plt.show(block=True)
#     plt.show()
#     dt = input('Enter amount to shift time by: ')
#     dt = int(dt)
#     if dt is 0:
#         done = True
