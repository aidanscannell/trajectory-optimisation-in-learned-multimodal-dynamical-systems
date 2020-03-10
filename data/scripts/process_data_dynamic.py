import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button

from parse_csv import (parse_traj_csv, parse_traj_csv_by_empty_line,
                       parse_vicon_csv)

# position of sheet
global x_alpha, y_alpha
x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 10
y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 10


def process_data(current_working_dir,
                 t_cut=20,
                 start_idxs_filename=None,
                 model_inputs_filename=None,
                 find_start_idx=False,
                 plot_all_trials_quiver=False,
                 plot_each_trial=False,
                 save_pdfs=True):

    # process data from vicon system
    vicon_data, start_pos = parse_vicon_csv(folder_name=current_working_dir,
                                            freq_hz=100)

    # process data from tello
    # tello_data = parse_traj_csv('traj.csv', time_interval=9)
    # tello_data = parse_traj_csv(folder_name + "/traj.csv", time_interval=9)
    tello_data = parse_traj_csv_by_empty_line(current_working_dir +
                                              "/traj2.csv")

    trials = list(map(lambda pos: int(pos) - 1, start_pos))
    print("\nTrials: ", str(trials))
    print("Start positions: " + str(start_pos))

    if find_start_idx is False:
        start_idxs = np.load(start_idxs_filename)['x']
    else:
        start_idxs = find_start_and_save(vicon_data,
                                         tello_data,
                                         t_cut,
                                         filename="start_idxs.npz",
                                         cwd=current_working_dir,
                                         trials=[0, 1, 2, 3])

    print('\nStart indexs being used: ', start_idxs)

    # process each trial to get all info wanted for plotting and training model
    p_data = []
    for trial in trials:
        p_data_trial, model_input_trial, model_output_trial = process_trial(
            vicon_data, tello_data, trial, int(start_idxs[trial]))
        p_data.append(p_data_trial)
        if 'model_inputs' not in locals():
            model_input = model_input_trial
            model_output = model_output_trial
        else:
            model_input = np.concatenate([model_input, model_input_trial])
            model_output = np.concatenate([model_output, model_output_trial])

    # np.savez(model_inputs_filename, x=model_input, y=model_output)

    for trial in trials:
        plot_z(p_data[trial], trial)
        print('here')

    if save_pdfs is True:
        cwd_split = re.split('/csv/', cwd)
        start_idxs_folder_name = cwd_split[0] + "/npz/" + cwd_split[1]
        # start_idxs_folder_name = re.split('csv/', folder_name)
        # start_idxs_folder_name = "../npz/" + start_idxs_folder_name[1]
        if not os.path.isdir(start_idxs_folder_name):
            os.makedirs(start_idxs_folder_name)
        for trial in trials:
            plt.figure()
            plot_vicon_with_quiver(p_data[trial])
            plt.savefig(start_idxs_folder_name + "/vicon_with_quiver_trial_" +
                        str(trial) + ".pdf",
                        transparent=True,
                        bbox_inches='tight')
        plt.figure()
        plt.quiver(model_input[:, 0], model_input[:, 1], model_output[:, 0],
                   model_output[:, 1])
        plt.plot(x_alpha, y_alpha)
        plt.savefig(start_idxs_folder_name + "/quiver_all_trials.pdf",
                    transparent=True,
                    bbox_inches='tight')

    if plot_all_trials_quiver is True:
        # visualise data using quiver plot
        plt.figure()
        plt.quiver(model_input[:, 0], model_input[:, 1], model_output[:, 0],
                   model_output[:, 1])
        plt.plot(x_alpha, y_alpha)
        plt.show(block=True)

    if plot_each_trial is True:
        for trial in trials:
            plt.figure()
            plot_vicon_with_quiver(p_data[trial])
            plt.show(block=True)


def plot_z(p_data, trial):
    plt.plot(p_data['time_vicon'], p_data['x_vicon'], zorder=4)
    plt.plot(p_data['time_vicon'], p_data['y_vicon'], zorder=4)
    plt.plot(p_data['time_vicon'], p_data['z_vicon'], zorder=4)

    plt.scatter(p_data['time_vicon'][p_data['test_point_idxs']],
                p_data['x_vicon'][p_data['test_point_idxs']],
                zorder=4)

    plt.scatter(p_data['time_vicon'][p_data['test_point_idxs']],
                p_data['y_vicon'][p_data['test_point_idxs']],
                zorder=4)

    plt.scatter(p_data['time_vicon'][p_data['test_point_idxs']],
                p_data['z_vicon'][p_data['test_point_idxs']],
                zorder=4)
    # plt.plot(p_data['x_tello'] + p_data['x_vicon'][0],
    #          p_data['y_tello'] + p_data['y_vicon'][0],
    #          label=trial)
    # plt.scatter(p_data['x_vicon_at_test_points'],
    #             p_data['y_vicon_at_test_points'],
    #             marker='x',
    #             c='k',
    #             zorder=10)
    # plt.quiver(p_data['x_vicon_at_test_points'],
    #            p_data['y_vicon_at_test_points'],
    #            p_data['dx_tello'],
    #            p_data['dy_tello'],
    #            angles='xy',
    #            scale_units='xy',
    #            width=0.001,
    #            scale=1,
    #            zorder=10)
    # for i in range(p_data['x_vicon_at_test_points'].shape[0]):
    # plt.annotate(i, (p_data['time'][i], p_data['time'][i]), zorder=2)
    # for i in range(p_data['x_vicon_at_test_points'].shape[0]):
    #     plt.annotate(i, (p_data['x_vicon_at_test_points'][i],
    #                      p_data['y_vicon_at_test_points'][i]),
    #                  zorder=2)

    # plt.quiver(p_data['x_vicon_at_test_points'][:-1] + p_data['dx_tello'][:-1],
    #            p_data['y_vicon_at_test_points'][:-1] + p_data['dy_tello'][:-1],
    #            p_data['dx'][1:],
    #            p_data['dy'][1:],
    #            angles='xy',
    #            scale_units='xy',
    #            width=0.001,
    #            color='r',
    #            scale=1,
    #            zorder=17)

    # plt.plot(x_alpha, y_alpha, zorder=8)
    plt.savefig('/Users/aidanscannell/python-projects/BMNSVGP/images/xyz' +
                str(trial) + '.pdf')
    plt.show(block=True)


# functions for interactively finding the starting index for vicon data
def onpick(event):
    global idx
    idx = event.ind
    plt.close()
    return True


def find_starting_index(vicon_data, trial, t_cut=20):
    """
    t_cut: set time index to find max z at beginning
    """
    time_vicon = vicon_data[trial]['time']
    x_vicon = vicon_data[trial]['x']
    y_vicon = vicon_data[trial]['y']
    z_vicon = vicon_data[trial]['z']

    high_idx = list(map(lambda i: i > t_cut, time_vicon)).index(True)

    # visualise starting point to make sure it is correct
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select start of trial')
    ax.plot(time_vicon[:high_idx], x_vicon[:high_idx], label='x', picker=1)
    ax.plot(time_vicon[:high_idx], y_vicon[:high_idx], label='y', picker=1)
    plt.legend()
    cid = fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block=True)
    start_idx = idx[-1]
    return start_idx


def process_trial(vicon_data, tello_data, trial, start_idx):
    time_vicon = vicon_data[trial]['time']
    time_vicon = time_vicon - time_vicon[start_idx]
    time_vicon = time_vicon[start_idx:]
    x_vicon = vicon_data[trial]['x'][start_idx:]
    y_vicon = vicon_data[trial]['y'][start_idx:]
    z_vicon = vicon_data[trial]['z'][start_idx:]
    rz_vicon = vicon_data[trial]['rz'][start_idx:]

    print('herherher')
    print(vicon_data[trial]['start_pos'])
    print(trial)
    tello_startp = int(vicon_data[trial]['start_pos']) - 1

    time_tello = tello_data[tello_startp]['time']
    time_tello = time_tello - time_tello[0]

    idx = [
        list(map(lambda t_v: t_v >= t_t, time_vicon)).index(True)
        for t_t in time_tello
    ]

    rz_vicon_at_test_points = rz_vicon[idx]

    x_tello, y_tello, z_tello, dx_tello, dy_tello, dz_tello = calc_tello_traj(
        tello_data, rz_vicon_at_test_points, tello_startp)

    dx, dy, dz = calc_error(x_vicon[idx], y_vicon[idx], z_vicon[idx], dx_tello,
                            dy_tello, dz_tello)

    trial_data = {}
    trial_data['test_point_idxs'] = idx
    trial_data['dx'] = dx
    trial_data['dy'] = dy
    trial_data['dz'] = dz

    trial_data['time_vicon'] = time_vicon
    trial_data['x_vicon'] = x_vicon
    trial_data['y_vicon'] = y_vicon
    trial_data['x_vicon_at_test_points'] = x_vicon[idx]
    trial_data['y_vicon_at_test_points'] = y_vicon[idx]
    trial_data['z_vicon_at_test_points'] = z_vicon[idx]
    trial_data['z_vicon'] = z_vicon
    trial_data['rz_vicon'] = rz_vicon

    trial_data['time_tello'] = time_tello
    trial_data['x_tello'] = x_tello
    trial_data['y_tello'] = y_tello
    trial_data['z_tello'] = z_tello
    trial_data['dx_tello'] = dx_tello
    trial_data['dy_tello'] = dy_tello
    trial_data['dz_tello'] = dz_tello

    model_input = np.stack([x_vicon[idx], y_vicon[idx]]).T
    model_output = np.stack([dx, dy, dz]).T

    return trial_data, model_input, model_output


def calc_tello_traj(tello_data, rz_vicon_at_test_points, startp):

    tello_xyz_matrix = np.stack([
        tello_data[startp]['x'], tello_data[startp]['y'],
        tello_data[startp]['z']
    ],
                                axis=1)
    tello_xyz_matrix = tello_xyz_matrix.reshape(-1, 1, 3)
    R = np.zeros([rz_vicon_at_test_points.shape[0], 3, 3])
    # print('herer')
    # print(rz_vicon_at_test_points.shape)
    for i, (c, s) in enumerate(
            zip(np.cos(rz_vicon_at_test_points),
                np.sin(rz_vicon_at_test_points))):
        R[i, ...] = np.array([[-c, -s, 0], [s, -c, 0], [0, 0, 1]])
    rotated_tello_xyz = tello_xyz_matrix @ R
    rotated_tello_xyz = np.squeeze(rotated_tello_xyz.reshape(-1, 3))

    x_tello = np.zeros([rotated_tello_xyz[:, 0].shape[0], 1])
    y_tello = np.zeros([rotated_tello_xyz[:, 1].shape[0], 1])
    z_tello = np.zeros([rotated_tello_xyz[:, 2].shape[0], 1])
    for i in range(0, x_tello.shape[0] - 1):
        x_tello[i + 1, 0] = x_tello[i, 0] + rotated_tello_xyz[i, 0]
        y_tello[i + 1, 0] = y_tello[i, 0] + rotated_tello_xyz[i, 1]
        z_tello[i + 1, 0] = z_tello[i, 0] + rotated_tello_xyz[i, 2]

    dx_tello = rotated_tello_xyz[:, 0]
    dy_tello = rotated_tello_xyz[:, 1]
    dz_tello = rotated_tello_xyz[:, 1]

    return x_tello, y_tello, z_tello, dx_tello, dy_tello, dz_tello


def calc_error(x_vicon_at_test_points, y_vicon_at_test_points,
               z_vicon_at_test_points, dx_tello, dy_tello, dz_tello):
    """ Calculate dx and dy at each test point """
    dx = np.zeros(len(x_vicon_at_test_points))
    dy = np.zeros(len(y_vicon_at_test_points))
    dz = np.zeros(len(z_vicon_at_test_points))
    for i in range(1, dx.shape[0] - 1):
        dx[i] = x_vicon_at_test_points[i] - x_vicon_at_test_points[
            i - 1] - dx_tello[i - 1]
        dy[i] = y_vicon_at_test_points[i] - y_vicon_at_test_points[
            i - 1] - dy_tello[i - 1]
        dz[i] = z_vicon_at_test_points[i] - z_vicon_at_test_points[
            i - 1] - dz_tello[i - 1]
    # for i in range(1, dx.shape[0] - 1):
    #     dx[i - 1] = x_vicon_at_test_points[i] - x_vicon_at_test_points[
    #         i - 1] - dx_tello[i - 1]
    #     dy[i - 1] = y_vicon_at_test_points[i] - y_vicon_at_test_points[
    #         i - 1] - dy_tello[i - 1]
    return dx, dy, dz


def plot_vicon_with_quiver(p_data):
    plt.plot(p_data['x_vicon'], p_data['y_vicon'], zorder=4)
    # plt.plot(p_data['x_tello'] + p_data['x_vicon'][0],
    #          p_data['y_tello'] + p_data['y_vicon'][0],
    #          label=trial)
    plt.scatter(p_data['x_vicon_at_test_points'],
                p_data['y_vicon_at_test_points'],
                marker='x',
                c='k',
                zorder=10)
    plt.quiver(p_data['x_vicon_at_test_points'],
               p_data['y_vicon_at_test_points'],
               p_data['dx_tello'],
               p_data['dy_tello'],
               angles='xy',
               scale_units='xy',
               width=0.001,
               scale=1,
               zorder=10)
    for i in range(p_data['x_vicon_at_test_points'].shape[0]):
        plt.annotate(i, (p_data['x_vicon_at_test_points'][i],
                         p_data['y_vicon_at_test_points'][i]),
                     zorder=2)

    plt.quiver(p_data['x_vicon_at_test_points'][:-1] + p_data['dx_tello'][:-1],
               p_data['y_vicon_at_test_points'][:-1] + p_data['dy_tello'][:-1],
               p_data['dx'][1:],
               p_data['dy'][1:],
               angles='xy',
               scale_units='xy',
               width=0.001,
               color='r',
               scale=1,
               zorder=17)

    plt.plot(x_alpha, y_alpha, zorder=8)


class find_start_index:
    def __init__(self, vicon_data, tello_data, trial, t_cut):
        self.vicon_data = vicon_data
        self.tello_data = tello_data
        self.trial = trial
        self.t_cut = t_cut

    def on_click_accept(self, event):
        plt.close()

    def on_click_retry(self, event):
        plt.close()
        self.calibrate_idxs()

    def calibrate_idxs(self):
        global start_idx
        start_idx = find_starting_index(self.vicon_data,
                                        self.trial,
                                        t_cut=self.t_cut)
        print('Initial calibrate idx: ', start_idx)
        p_data, _, _ = process_trial(self.vicon_data, self.tello_data,
                                     self.trial, start_idx)
        plt.figure()
        plot_vicon_with_quiver(p_data)
        ax_accept = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_retry = plt.axes([0.81, 0.05, 0.1, 0.075])
        accept_button = Button(ax_accept, 'Accept')
        accept_button.on_clicked(self.on_click_accept)
        retry_button = Button(ax_retry, 'Retry')
        retry_button.on_clicked(self.on_click_retry)
        plt.show(block=True)


def find_start_and_save(vicon_data,
                        tello_data,
                        t_cut,
                        filename="start_idxs.npz",
                        cwd=None,
                        trials=[0, 1, 2, 3]):
    """ find start index for each trial """
    start_idxs = np.zeros(len(trials))
    for trial in trials:
        print('\nFinding start index for trial ' + str(trial + 1) + '...')
        finder = find_start_index(vicon_data, tello_data, trial, t_cut)
        finder.calibrate_idxs()
        print('Trial ' + str(trial) + ' starting index: ', start_idx)
        start_idxs[trial] = int(start_idx)
    cwd_split = re.split('/csv/', cwd)
    start_idxs_folder_name = cwd_split[0] + "/npz/" + cwd_split[1]
    # start_idxs_folder_name = re.split('csv/', folder_name)
    # start_idxs_folder_name = "../npz/" + start_idxs_folder_name[1]
    if not os.path.isdir(start_idxs_folder_name):
        os.makedirs(start_idxs_folder_name)
    np.savez(start_idxs_folder_name + "/" + filename, x=start_idxs)
    # for trial in trials:
    #     plt.figure()
    #     plot_vicon_with_quiver(p_data[trial])
    #     plt.savefig(start_idxs_folder_name + "/vicon_with_quiver.pdf",
    #                 transparent=True,
    #                 bbox_inches='tight')
    # plt.show(block=True)
    return start_idxs


if __name__ == "__main__":
    cwd = os.getcwd()
    # folder_name = '../csv/26nov/1'
    folder_name = cwd
    print(folder_name)
    cwd_split = re.split('/csv/', cwd)
    print(cwd)
    start_idxs_filename = cwd_split[0] + "/npz/" + cwd_split[
        1] + "/start_idxs.npz"
    model_inputs_filename = cwd_split[0] + "/npz/" + cwd_split[
        1] + "/model_inputs.npz"
    # folder_name = cwd_split[0] + "/npz/" + cwd_split[1] + "/start_idxs.npz"
    print(folder_name)
    process_data(
        cwd,
        t_cut=50,
        start_idxs_filename=start_idxs_filename,
        # model_inputs_filename=model_inputs_filename,
        model_inputs_filename=model_inputs_filename)
    # find_start_idx=True,
    # plot_all_trials_quiver=True,
    # plot_each_trial=True,
    # plot_each_trial=True)
    # save_pdfs=True)
