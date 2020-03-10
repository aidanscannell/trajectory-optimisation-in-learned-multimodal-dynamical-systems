import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# position of sheet
global x_alpha, y_alpha
x_alpha = np.array([-345, -332, 1836, 1834, -345]) / 1000
y_alpha = np.array([-551, 1954, 1943, -586, -551]) / 1000

# rotation and translation of start positions
angles = [np.pi / 2, np.pi, 0, -np.pi / 2]
angles = [np.pi / 2, np.pi, 0, 0]

# angles = [np.pi, np.pi, np.pi, np.pi]  # 20dec 1
translations_x = np.array([-1981, -1905, 3086, 3035]) / 1000
translations_y = np.array([-2435, 2556, -2576, 2628]) / 1000


def parse_vicon_csv(folder_name=None,
                    model_inputs_filename='../npz/model_data.npz'):
    file_name_list = glob.glob(folder_name + '/*vicon-data*.csv')
    list_dicts = []
    start_pos = []
    for i, file_name in enumerate(file_name_list):
        model_input_trial, model_output_trial = parse_single_csv(file_name)
        if 'model_input' not in locals():
            model_input = model_input_trial[10:-10]
            model_output = model_output_trial[10:-10]
        else:
            model_input = np.concatenate(
                [model_input, model_input_trial[10:-10]])
            model_output = np.concatenate(
                [model_output, model_output_trial[10:-10]])

    # remove data where tello is flying along y axis
    # x2_low = -3.
    x2_low = -2.5
    x2_high = 2.
    x1_high = 3.
    mask_1 = model_input[:, 1] > x2_low
    mask_3 = model_input[:, 1] < x2_high
    mask_2 = model_input[:, 0] < x1_high
    mask = mask_1 & mask_3 & mask_2
    model_input = model_input[mask, :]
    model_output = model_output[mask, :]

    print('final')
    print(model_input.shape)
    print(model_output.shape)
    # np.savez(model_inputs_filename, x=model_input, y=model_output)

    dxyz = model_output[:, 3]
    x = model_input[:, 0]
    y = model_input[:, 1]
    # fig, ax2 = plt.subplots(nrows=1)
    fig = plt.figure(figsize=(12, 12))
    plt.quiver(x,
               y,
               dxyz,
               np.zeros([*dxyz.shape]),
               angles='xy',
               scale_units='xy',
               width=0.001,
               scale=1,
               zorder=10,
               color='white')
    plt.axis('off')
    save_name = 'images/quiver.png'
    plt.savefig(save_name, transparent=True, bbox_inches='tight')
    # ax2.tricontour(x, y, dxyz, levels=14, linewidths=0.5, colors='k')
    # cntr2 = ax2.tricontourf(x, y, dxyz, levels=14, cmap="RdBu_r")
    # plt.scatter(x, y, color='k', marker='x')
    plt.show()


def rotate_and_translate(start_pos, data_dict):
    tello_pos = np.stack(
        [data_dict['tello_x'], data_dict['tello_y'], data_dict['tello_z']],
        axis=1)
    theta = angles[start_pos - 1]
    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotated_tello_xyz = tello_pos @ R
    rotated_tello_xyz[:, 0] *= -1
    rotated_tello_xyz[:, 0] += data_dict['vicon_x'][0]
    rotated_tello_xyz[:, 1] += data_dict['vicon_y'][0]
    rotated_tello_xyz[:, 2] += data_dict['vicon_z'][0]
    data_dict['vicon_rz'] -= data_dict['vicon_rz'][0]
    # data_dict['vicon_rz'] = -data_dict['vicon_rz']

    return rotated_tello_xyz[:, 0], rotated_tello_xyz[:, 1], data_dict


def calc_error(data_dict, step, tello_x, tello_y):
    N = round(tello_x.shape[0] / step)
    dx = np.zeros(N + 1)
    dy = np.zeros(N + 1)
    dz = np.zeros(N + 1)
    dxyz = np.zeros(N + 1)
    drz = np.zeros(N + 1)
    n_steps_array = np.zeros(N + 1)
    # dx_sum = 0
    # dy_sum = 0
    # dz_sum = 0
    counter = 0
    tello_zero_idx = []
    n_steps = 1
    for i in range(1, tello_x.shape[0] - 1)[::step]:
        diff_rz = data_dict['vicon_rz'][i] - data_dict['tello_rz'][i]
        drz[counter] = abs(diff_rz)
        if int(tello_x[i] * 1) == int(data_dict['vicon_x'][0] * 1) and int(
                tello_y[i] * 1) == int(data_dict['vicon_y'][0] * 1):
            # if int(tello_x[i] * 100) == int(data_dict['vicon_x'][0] * 100) or int(
            #         tello_x[i - step] * 100) == int(data_dict['vicon_x'][0] * 100):
            # print('ZERO')
            # print('i: %i' % i)
            tello_zero_idx.append(i)
            n_steps += 1
            # print(n_steps)
        else:
            # print('NON ZERO')
            dx[counter] = (data_dict['vicon_x'][i] -
                           data_dict['vicon_x'][i - step * n_steps])
            #- (tello_x[i] - tello_x[i - step * n_steps])) / n_steps
            dy[counter] = (data_dict['vicon_y'][i] -
                           data_dict['vicon_y'][i - step * n_steps])
            #- (tello_y[i] - tello_y[i - step * n_steps])) / n_steps
            dz[counter] = (data_dict['vicon_z'][i] -
                           data_dict['vicon_z'][i - step * n_steps])

            n_steps = 1
        n_steps_array[counter] = int(n_steps)
        counter += 1

    return dx, dy, dz, n_steps_array, tello_zero_idx, dxyz, drz


def parse_single_csv(file_name):
    print('Parsing: %s' % file_name)
    data = pd.read_csv(file_name, index_col=0)
    data_np = data.to_numpy()
    # data_np = data.to_numpy()[800:-400]
    # data_np = data_np[500:]
    print('here')
    print(data_np.shape)
    # data_np = data_np[100:]
    # print(data_np.shape)

    data_dict = {}
    for idx, col in enumerate(data.columns):
        data_dict[col] = data_np[:, idx]

    if data_dict['vicon_x'][0] > 0 and data_dict['vicon_y'][0] > 0:
        start_pos = 4
    elif data_dict['vicon_x'][0] < 0 and data_dict['vicon_y'][0] > 0:
        start_pos = 2
    elif data_dict['vicon_x'][0] > 0 and data_dict['vicon_y'][0] < 0:
        start_pos = 3
    elif data_dict['vicon_x'][0] < 0 and data_dict['vicon_y'][0] < 0:
        start_pos = 1
    print('start position: %i' % start_pos)

    start_idx = 0
    for i in range(data_np.shape[0]):
        if data_dict['vicon_x'][i] < -1.8 and data_dict['vicon_y'][i] > -1:
            start_idx = i
            # print('x: ', data_dict['vicon_x'][i])
            # print('y: ', data_dict['vicon_y'][i])
            break

    data_dict = {}
    for idx, col in enumerate(data.columns):
        data_dict[col] = data_np[start_idx:, idx]

    tello_x, tello_y, data_dict = rotate_and_translate(start_pos, data_dict)

    data_dict['tello_x'] = tello_x
    data_dict['tello_y'] = tello_y
    step = 20

    # step = 40
    # step = 50
    dx, dy, dz, n_steps_array, tello_zero_idx, dxyz, drz = calc_error(
        data_dict, step, tello_x, tello_y)

    if data_dict['vicon_y'][0::step].shape != dy.shape:
        print('different shapes...')
        diff_x = data_dict['vicon_x'][0::step].shape[0] - dx.shape[0]
        diff_y = data_dict['vicon_y'][0::step].shape[0] - dy.shape[0]
        diff_z = data_dict['vicon_z'][0::step].shape[0] - dz.shape[0]
        print(diff_x)
        print(diff_y)
        dx = dx[:diff_x]
        dy = dy[:diff_y]
        dz = dz[:diff_z]
        dxyz = dxyz[:diff_z]
        drz = drz[:diff_z]
        # dx = dx[-diff_x:]
        # dy = dy[-diff_y:]
        # dx = dx[:-1]
        # dy = dy[:-1]

    if start_pos == 1:
        dxyz = dx
    elif start_pos == 2:
        dxyz = dx
    elif start_pos == 3:
        dxyz = dx
    elif start_pos == 4:
        dxyz = dy

    model_input = np.stack(
        [data_dict['vicon_x'][0::step], data_dict['vicon_y'][0::step]]).T
    model_output = np.stack([dx, dy, dz, dxyz]).T
    print(model_input.shape)
    print(model_output.shape)

    # plt.scatter(
    #     data_dict['vicon_x'][0::step][30:-30],
    #     data_dict['vicon_y'][0::step][30:-30],
    #     marker='x',
    #     # mew=1,
    #     s=10,
    #     linewidth=0.5,
    #     color='k',
    #     alpha=0.6,
    #     zorder=5)

    # for i in range(n_steps_array.shape[0] - 1):
    #     # print(i)
    #     if n_steps_array[i] != 1 and i > 100:
    #         n_steps = int(n_steps_array[i])
    #         print(n_steps)
    #         plt.scatter(data_dict['vicon_x'][::step][i - n_steps],
    #                     data_dict['vicon_y'][::step][i - n_steps],
    #                     marker='x',
    #                     color='k')
    #         plt.scatter(tello_x[::step][i - n_steps],
    #                     tello_y[::step][i - n_steps],
    #                     marker='x')
    # plt.show()
    # for i in range(0, tello_x.shape[0])[::step]:

    # plt.annotate(
    #     n_steps_array[i],
    #     # plt.annotate(n_steps_array[i],
    #     xy=(tello_x[::step][i - 1], tello_y[::step][i - 1]))
    # xy=(tello_x[::step][i - n_steps_array[i]],
    # tello_y[::step][i - n_steps_array[i]]))

    # time = np.arange(0, drz.shape[0])
    # print(time.shape)
    # plt.plot(time, drz)
    # plt.show()

    x = data_dict['vicon_x'][0::step]
    y = data_dict['vicon_y'][0::step]
    z = dxyz
    # z = drz
    # fig, ax2 = plt.subplots(nrows=1)
    # ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    # cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
    # fig.colorbar(cntr2, ax=ax2)
    # ax2.plot(x, y, 'ko', ms=3)
    # ax2.set(xlim=(-2, 2), ylim=(-2, 2))
    # ax2.set_title('tricontour (%d points)' % npts)
    # plt.subplots_adjust(hspace=0.5)
    # plt.show()

    # plt.scatter(
    #     tello_x[10:],
    #     tello_y[10:],
    #     marker='x',
    #     # mew=1,
    #     s=10,
    #     linewidth=0.5,
    #     color='k',
    #     alpha=0.6,
    #     zorder=5)
    # fig, ax2 = plt.subplots(nrows=1)
    # ax2.tricontour(x, y, dxyz, levels=14, linewidths=0.5, colors='k')
    # cntr2 = ax2.tricontourf(x, y, dxyz, levels=14, cmap="RdBu_r")
    # plt.show()
    fig = plt.figure(figsize=(12, 12))
    plt.quiver(x,
               y,
               dxyz,
               np.zeros([*dxyz.shape]),
               angles='xy',
               scale_units='xy',
               width=0.001,
               scale=1,
               zorder=10,
               color='white')
    # plt.quiver(data_dict['vicon_x'][0::step],
    #            data_dict['vicon_y'][0::step],
    #            dx,
    #            dy,
    #            angles='xy',
    #            scale_units='xy',
    #            width=0.001,
    #            scale=1,
    #            zorder=10)

    # plt.plot(data_dict['tello_x'],
    #          data_dict['tello_y'],
    #          label='tello original')
    # mask = np.ones(tello_x.shape, bool)
    # mask[tello_zero_idx] = False
    # plt.plot(tello_x[mask], tello_y[mask], label='tello transformed')
    # plt.plot(tello_x, tello_y, label='tello transformed')
    # plt.annotate('start',
    #              xy=(data_dict['tello_x'][0], data_dict['tello_y'][0]))
    # plt.annotate('start',
    #              xy=(data_dict['vicon_x'][0], data_dict['vicon_y'][0]))
    plt.plot(data_dict['vicon_x'],
             data_dict['vicon_y'],
             label='vicon',
             color="darkmagenta")
    # plt.scatter(data_dict['vicon_x'][::step],
    #             data_dict['vicon_y'][::step],
    #             marker='x',
    #             color='k',
    #             label='vicon')
    # plt.scatter(tello_x[::step],
    #             tello_y[::step],
    #             marker='x',
    #             color='k',
    #             label='tello')
    # plt.plot(-data_dict['vicon_y'],
    #          data_dict['vicon_x'],
    #          label='vicon rotated')
    # plt.plot(x_alpha, y_alpha)
    plt.axis('off')
    # plt.xlim(-10, 12)
    # plt.legend()
    save_name = 'images/trajectory.png'
    plt.savefig(save_name, transparent=True, bbox_inches='tight')
    plt.show(block=True)
    return model_input, model_output


filename = '../csv/closed-loop/19dec/'

if __name__ == "__main__":
    # cwd = os.getcwd()
    # folder_name = '../csv/closed-loop/19dec'
    # folder_name = '../csv/closed-loop/20dec/rz'
    folder_name = '../csv/turbulence/21-feb/start-pos1/test'
    folder_name = '../csv/turbulence/21-feb/start-pos1'
    folder_name = '../csv/turbulence/25-feb/fan-fixed-setting-2-center'
    folder_name = '../csv/turbulence/25-feb/fan-fixed-setting-2-center/subset'
    folder_name = '../csv/turbulence/27-feb/half-lengthscale'
    # folder_name = '../csv/closed-loop/20dec'
    # folder_name = cwd
    print(folder_name)
    # cwd_split = re.split('loop/', cwd)
    parse_vicon_csv(
        folder_name,
        model_inputs_filename=
        '../npz/turbulence/model_data_fan_fixed_middle_short_lengthscale.npz')
