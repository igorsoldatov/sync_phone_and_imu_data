import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
from numpy import genfromtxt
from scipy.signal import butter, filtfilt


def apply_filter_df(df, N, Wn):
    b, a = butter(N, Wn)

    df[' ax[m/s^2]'] = filtfilt(b, a, df[' ax[m/s^2]'])
    df[' ay[m/s^2]'] = filtfilt(b, a, df[' ay[m/s^2]'])
    df[' az[m/s^2]'] = filtfilt(b, a, df[' az[m/s^2]'])

    df[' gx[rad/s]'] = filtfilt(b, a, df[' gx[rad/s]'])
    df[' gy[rad/s]'] = filtfilt(b, a, df[' gy[rad/s]'])
    df[' gz[rad/s]'] = filtfilt(b, a, df[' gz[rad/s]'])


def apply_filter_np(N, Wn, data):
    b, a = butter(N, Wn)
    data = filtfilt(b, a, data)
    return data


def show_data(file_phone, file_imu, sync_params, filter=False):
    fig = plt.figure()
    if filter:
        fig.set_size_inches(30., 18.)
        dpi = 600
    else:
        fig.set_size_inches(12., 7.)
        dpi = 300

    index_sync = sync_params['index_sync']
    phone_acc_start = index_sync['Accelerometer']['phone']['start']
    phone_acc_end = index_sync['Accelerometer']['phone']['end']
    imu_acc_start = index_sync['Accelerometer']['imu']['start']
    imu_acc_end = index_sync['Accelerometer']['imu']['end']
    phone_gyr_start = index_sync['Gyroscope']['phone']['start']
    phone_gyr_end = index_sync['Gyroscope']['phone']['end']
    imu_gyr_start = index_sync['Gyroscope']['imu']['start']
    imu_gyr_end = index_sync['Gyroscope']['imu']['end']
    name = f"{sync_params['phone_file']}-{sync_params['imu_file']}"

    df_phone = pd.read_csv(file_phone)
    df_imu = pd.read_csv(file_imu)

    if filter:
        apply_filter_df(df_phone, 3, 0.02)
        apply_filter_df(df_imu, 3, 0.02)

    gir1 = np.asarray(np.sqrt(df_phone[' gx[rad/s]'] ** 2 + df_phone[' gy[rad/s]'] ** 2 + df_phone[' gz[rad/s]'] ** 2))
    gir2 = np.asarray(np.sqrt(df_imu[' gx[rad/s]'] ** 2 + df_imu[' gy[rad/s]'] ** 2 + df_imu[' gz[rad/s]'] ** 2))

    acc1 = np.asarray(np.sqrt(df_phone[' ax[m/s^2]'] ** 2 + df_phone[' ay[m/s^2]'] ** 2 + df_phone[' az[m/s^2]'] ** 2))
    acc2 = np.asarray(np.sqrt(df_imu[' ax[m/s^2]'] ** 2 + df_imu[' ay[m/s^2]'] ** 2 + df_imu[' az[m/s^2]'] ** 2))

    # Accelerometer:
    # diff2: 137461, start1: 12283, end1: 134905, start2: 0, end2: 122622, div: 122622

    if filter:
        acc1 = apply_filter_np(3, 0.01, acc1)
        acc2 = apply_filter_np(3, 0.01, acc2)

    indexes = [*range(len(acc1[phone_acc_start:phone_acc_end]))]

    ax = plt.axes()
    ax.plot(indexes, acc1[phone_acc_start:phone_acc_end], label='phone')
    ax.plot(indexes, acc2[imu_acc_start:imu_acc_end], label='imu')
    plt.title(f"")
    plt.legend(framealpha=1, frameon=True)
    # plt.show()
    plt.savefig(f'./preview/acc-{name}-filer={filter}.png', dpi=dpi)

    # Gyroscope:
    # diff2: 59799, start1: 89945, end1: 149744, start2: 0, end2: 59799, div: 59799

    if filter:
        gir1 = apply_filter_np(3, 0.01, gir1)
        gir2 = apply_filter_np(3, 0.01, gir2)

    indexes = [*range(len(gir1[phone_gyr_start:phone_gyr_end]))]

    ax = plt.axes()
    ax.plot(indexes, gir1[phone_gyr_start:phone_gyr_end], label='phone')
    ax.plot(indexes, gir2[imu_gyr_start:imu_gyr_end], label='imu')
    plt.title(f"")
    plt.legend(framealpha=1, frameon=True)
    # plt.show()
    plt.savefig(f'./preview/gyr-{name}-filer={filter}.png', dpi=dpi)


