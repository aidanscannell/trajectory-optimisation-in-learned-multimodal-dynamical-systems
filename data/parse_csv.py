import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_traj_csv(file_name, time_interval=10):
    data_tello = pd.read_csv(file_name, index_col=0)

    time_tello = data_tello.index
    data_tello = data_tello.to_numpy()
    x_tello = data_tello[:, 0]
    y_tello = data_tello[:, 1]
    z_tello = data_tello[:, 2]

    idxs = []
    for i in range(1, data_tello.shape[0]):
        if time_tello[i] - time_tello[i - 1] > time_interval:
            idxs.append(i)

    print("Found " + str(len(idxs)) + " trials using time interval of " +
          str(time_interval) + "s")
    list_dicts = []
    for i in range(len(idxs)):
        if i == 0:
            low = 0
        else:
            low = idxs[i - 1]
        high = idxs[i]
        file_dict = {}
        file_dict['time'] = time_tello[low:high]
        file_dict['x'] = x_tello[low:high]
        file_dict['y'] = y_tello[low:high]
        file_dict['z'] = z_tello[low:high]
        list_dicts.append(file_dict)
        # plt.scatter(time_tello[low:high], x_tello[low:high])
        # plt.scatter(time_tello[low:high], y_tello[low:high])
        # plt.plot(time_tello[low:high], x_tello[low:high])
        # plt.plot(time_tello[low:high], y_tello[low:high])
        # plt.show()
        # input()
        # plt.close()

    return list_dicts


def parse_single_vicon_csv(file_name, freq=100):
    print('Parsing file: ' + file_name)
    data = pd.read_csv(file_name, header=2, index_col=0)
    for idx, col in enumerate(data.columns):
        if "Global Angle Tello_DC5B8B:Tello_DC5B8B" in col:
            i = idx
    cols = np.arange(i, i + 7).reshape(-1, 1)
    zero = np.array([[0]])
    cols = np.vstack([zero, cols])
    data_vicon = pd.read_csv(file_name,
                             header=3,
                             index_col=0,
                             usecols=cols.flatten(),
                             skiprows=[4])
    data = data_vicon.to_numpy()

    # set starting point to (0, 0, 0)
    # z_vicon = data[:, 6] - data[0, 6]
    # y_vicon = data[:, 5] - data[0, 5]
    # x_vicon = data[:, 4] - data[0, 4]
    x_vicon = data[:, 4] / 10
    y_vicon = data[:, 5] / 10
    z_vicon = data[:, 6] / 10

    # convert time into seconds
    print('herherhehre')
    print(data_vicon.index[0])
    time_vicon = data_vicon.index - data_vicon.index[0]
    time_vicon = np.arange(0, x_vicon.shape[0]) * (1 / 100)
    return time_vicon, x_vicon, y_vicon, z_vicon


def parse_vicon_csv(file_name, folder_name=None, freq=100):
    if folder_name is not None:
        file_name_list = glob.glob(folder_name + '/' + file_name)
        file_name_list.sort(key=os.path.getmtime)
        # print("\n".join(file_name_list))
    else:
        file_name_list = [file_name]
    list_dicts = []
    start_pos = []
    for i, file_name in enumerate(file_name_list):
        time, x, y, z = parse_single_vicon_csv(file_name, freq)
        file_dict = {}
        file_dict['time'] = time
        print(file_name)
        # if ('startp2' or 'startp3') in file_name:
        start_pos.append(file_name.split('startp', 1)[1].split('-', 1)[0])
        #     file_dict['x'] = y
        #     file_dict['y'] = x
        #     print("here")
        # else:
        #     print("here1")
        file_dict['x'] = x
        file_dict['y'] = y
        file_dict['z'] = z
        list_dicts.append(file_dict)
    return list_dicts, start_pos
