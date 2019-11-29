import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_traj_csv_by_empty_line(file_name):
    data_tello = pd.read_csv(file_name, index_col=0)

    time_tello = data_tello.index

    df = data_tello
    idxs = np.where(np.isnan(df.index))[0].tolist()
    print(idxs)

    data_tello = data_tello.to_numpy()
    x_tello = data_tello[:, 0]
    y_tello = data_tello[:, 1]
    z_tello = data_tello[:, 2]

    idxs.append(data_tello.shape[0])

    print("Found " + str(len(idxs)) + " trials using empty lines")
    list_dicts = []
    for i in range(len(idxs)):
        file_dict = {}
        high = idxs[i] - 1
        if i == 0:
            low = 0
        else:
            low = idxs[i - 1] + 2
        file_dict['time'] = time_tello[low:high]
        file_dict['x'] = x_tello[low:high]
        file_dict['y'] = y_tello[low:high]
        file_dict['z'] = z_tello[low:high]
        list_dicts.append(file_dict)
    return list_dicts


def parse_traj_csv(file_name, time_interval=10):
    data_tello = pd.read_csv(file_name, index_col=0)

    time_tello = data_tello.index

    idxs = []
    for i in range(1, data_tello.to_numpy().shape[0]):
        if time_tello[i] - time_tello[i - 1] > time_interval:
            idxs.append(i)

    df = data_tello
    idxs = np.where(np.isnan(df.index))[0].tolist()
    print(idxs)

    data_tello = data_tello.to_numpy()
    x_tello = data_tello[:, 0]
    y_tello = data_tello[:, 1]
    z_tello = data_tello[:, 2]
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
    return list_dicts


def parse_single_vicon_csv(file_name, freq_hz=100):
    print('Parsing file: ' + file_name)
    data = pd.read_csv(file_name, header=2, index_col=0)
    for idx, col in enumerate(data.columns):
        if "Global Angle Tello_DC5B8B:Tello_DC5B8B" in col:
            i = idx
        elif "Global Angle Tello_FCF6BF:Tello_FCF6BF" in col:
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

    rz_vicon = data[:, 3]
    x_vicon = data[:, 4] / 10
    y_vicon = data[:, 5] / 10
    z_vicon = data[:, 6] / 10

    # convert time into seconds
    time_vicon = data_vicon.index - data_vicon.index[0]
    time_vicon = np.arange(0, x_vicon.shape[0]) * (1 / 100)
    return time_vicon, x_vicon, y_vicon, z_vicon, rz_vicon


def parse_vicon_csv(folder_name=None, freq_hz=100):
    file_name_list = glob.glob(folder_name + '/*startp*.csv')
    # file_name_list.sort(key=os.path.getmtime)
    list_dicts = []
    start_pos = []
    for i, file_name in enumerate(file_name_list):
        time, x, y, z, rz = parse_single_vicon_csv(file_name, freq_hz)
        file_dict = {}
        file_dict['time'] = time
        file_dict['start_pos'] = file_name.split('startp', 1)[1].split('-',
                                                                       1)[0]
        start_pos.append(file_name.split('startp', 1)[1].split('-', 1)[0])
        file_dict['x'] = x
        file_dict['y'] = y
        file_dict['z'] = z
        file_dict['rz'] = rz
        list_dicts.append(file_dict)
    return list_dicts, start_pos
    # return list_dicts
