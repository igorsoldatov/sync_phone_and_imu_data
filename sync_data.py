import sys
import json
import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy import interpolate
from pyquaternion import Quaternion

from scipy.integrate import odeint
from os import listdir
from os import walk
from os.path import isdir, join, isfile, dirname
from scipy.signal import butter, filtfilt

from preview_data import *


def pre_w(df):
    # интерполяция угловых скоростей для численного решения уравнения пуассона
    T = df['Timestamp[nanosec]'] / 10 ** 9
    W = np.asarray([df[' gx[rad/s]'], df[' gy[rad/s]'], df[' gz[rad/s]']]).T
    tck1 = interpolate.splrep(T, W[:, 0], s=0)
    tck2 = interpolate.splrep(T, W[:, 1], s=0)
    tck3 = interpolate.splrep(T, W[:, 2], s=0)

    def w(t):
        ynew1 = interpolate.splev(t, tck1, der=0)
        ynew2 = interpolate.splev(t, tck2, der=0)
        ynew3 = interpolate.splev(t, tck3, der=0)
        return np.asarray([ynew1, ynew2, ynew3])

    return w


def create_ode(func_w):
    def poisson(Y, t):
        # система оде для уравнения пуассона
        dYdt = 0.5 * Quaternion(scalar=0, vector=func_w(t)) * Quaternion(Y)
        return dYdt.elements

    return poisson


def sync_and_toGSC(df1, df2):
    sync_params = {}

    # сначала применяется ФНЧ ко всем данным
    b, a = butter(3, 0.02)

    df1[' ax[m/s^2]'] = filtfilt(b, a, df1[' ax[m/s^2]'])
    df1[' ay[m/s^2]'] = filtfilt(b, a, df1[' ay[m/s^2]'])
    df1[' az[m/s^2]'] = filtfilt(b, a, df1[' az[m/s^2]'])

    df1[' gx[rad/s]'] = filtfilt(b, a, df1[' gx[rad/s]'])
    df1[' gy[rad/s]'] = filtfilt(b, a, df1[' gy[rad/s]'])
    df1[' gz[rad/s]'] = filtfilt(b, a, df1[' gz[rad/s]'])

    df2[' ax[m/s^2]'] = filtfilt(b, a, df2[' ax[m/s^2]'])
    df2[' ay[m/s^2]'] = filtfilt(b, a, df2[' ay[m/s^2]'])
    df2[' az[m/s^2]'] = filtfilt(b, a, df2[' az[m/s^2]'])

    df2[' gx[rad/s]'] = filtfilt(b, a, df2[' gx[rad/s]'])
    df2[' gy[rad/s]'] = filtfilt(b, a, df2[' gy[rad/s]'])
    df2[' gz[rad/s]'] = filtfilt(b, a, df2[' gz[rad/s]'])

    gir1 = np.asarray(np.sqrt(df1[' gx[rad/s]'] ** 2 + df1[' gy[rad/s]'] ** 2 + df1[' gz[rad/s]'] ** 2))
    gir2 = np.asarray(np.sqrt(df2[' gx[rad/s]'] ** 2 + df2[' gy[rad/s]'] ** 2 + df2[' gz[rad/s]'] ** 2))

    b, a = butter(3, 0.01)
    gir1 = filtfilt(b, a, gir1)
    b, a = butter(3, 0.01)
    gir2 = filtfilt(b, a, gir2)

    # Перебором находится время сдвига для синхронизации отдельно для акселерометра и ДУС
    maxerr = 10000000
    best_diff1, best_start1, best_end1, best_start2, best_end2, best_div = 0, 0, 0, 0, 0, 0
    for diff in tqdm(range(100, len(df1) + len(df2) - 100)):
        start1 = max(0, len(df1) - diff)
        end1 = min(len(df1), len(df1) + len(df2) - diff)
        start2 = max(0, diff - len(df1))
        end2 = min(diff, len(df2))
        div = (min(diff, len(df2)) - max(0, diff - len(df1)))

        err = np.linalg.norm(gir1[start1:end1] - gir2[start2:end2]) / div

        if err < maxerr:
            best_diff1, best_start1, best_end1, best_start2, best_end2, best_div = diff, start1, end1, start2, end2, div
            maxerr = err

    print("Gyroscope:")
    print(f"diff2: {best_diff1}, phone {best_start1}:{best_end1}, imu {best_start2}:{best_end2}, "
          f"div: {best_div}")
    sync_params['Gyroscope'] = {'phone': {'start': best_start1, 'end': best_end1},
                                'imu': {'start': best_start2, 'end': best_end2}}

    acc1 = np.asarray(np.sqrt(df1[' ax[m/s^2]'] ** 2 + df1[' ay[m/s^2]'] ** 2 + df1[' az[m/s^2]'] ** 2))
    acc2 = np.asarray(np.sqrt(df2[' ax[m/s^2]'] ** 2 + df2[' ay[m/s^2]'] ** 2 + df2[' az[m/s^2]'] ** 2))

    b, a = butter(3, 0.01)
    acc1 = filtfilt(b, a, acc1)
    b, a = butter(3, 0.01)
    acc2 = filtfilt(b, a, acc2)

    maxerr = 1000000
    best_diff2, best_start1, best_end1, best_start2, best_end2, best_div = 0, 0, 0, 0, 0, 0
    _diff, _start1, _end1, _start2, _end2, _div = [], [], [], [], [], []
    for diff in tqdm(range(100, len(df1) + len(df2) - 100)):
        start1 = max(0, len(df1) - diff)
        end1 = min(len(df1), len(df1) + len(df2) - diff)
        start2 = max(0, diff - len(df1))
        end2 = min(diff, len(df2))
        div = (min(diff, len(df2)) - max(0, diff - len(df1)))

        err = np.linalg.norm(acc1[start1:end1] - acc2[start2:end2]) / div

        if err < maxerr:
            best_diff2, best_start1, best_end1, best_start2, best_end2, best_div = diff, start1, end1, start2, end2, div
            maxerr = err

    print("Accelerometer:")
    print(f"diff2: {best_diff2}, phone {best_start1}:{best_end1}, imu {best_start2},{best_end2}, div: {best_div}")
    sync_params['Accelerometer'] = {'phone': {'start': best_start1, 'end': best_end1},
                                    'imu': {'start': best_start2, 'end': best_end2}}

    return sync_params


def main():
    print('input PATH of each file or from root (Y/N)')
    path = []
    flag_path = (str(input()) == 'Y')
    dfs = []
    if flag_path:
        print("input directory for information about sync:")
        cur_path = str(input())
        print("input number of files:")
        n = int(input())
        for i in range(n):
            print("input path to file:")
            path_c = str(input())
            if (isfile(path_c + "\\gyro_accel.csv")):
                path.append(path_c)
                df = pd.read_csv(path[-1] + "\\gyro_accel.csv")
                dfs.append(df)
    else:
        cur_path = dirname(sys.argv[0])
        print(cur_path)
        onlydirs = [d for d in listdir(cur_path) if isdir(join(cur_path, d))]
        for p in onlydirs:
            if (isfile(cur_path + '\\' + p + "\\gyro_accel.csv")):
                path.append(cur_path + '\\' + p)
                df = pd.read_csv(path[-1] + "\\gyro_accel.csv")
                dfs.append(df)

    times = []
    for i in range(len(dfs) - 1):
        time_sync, time_len = sync_and_toGSC(dfs[i], dfs[i + 1], path[i + 1])
        times.append([time_sync, time_len])
    _, time_len = sync_and_toGSC(dfs[-1], dfs[0], path[0])

    add_inf = []
    add_inf.append([path[0], time_len, '-'])
    j = 0
    for el in times:
        add_inf.append([path[j + 1], el[1], el[0]])
        j = j + 1
    add_inf = pd.DataFrame(add_inf)
    add_inf.columns = ["Path to current file", "Time length of current file [sec]",
                       "Sync time relative previous file [sec]"]
    add_inf.to_csv(cur_path + "\\sync_times.csv", encoding='utf-8')


def imu_preprocess(df):
    imu, server_time, arduino_time, ax, ay, az, gx, gy, gz, mx, my, mz = [], [], [], [], [], [], [], [], [], [], [], []

    def add(row):
        imu.append(row['imu'])
        server_time.append(row['server_time'])
        arduino_time.append(row['arduino_time'] * 1000000)
        ax.append(row['ax'])
        ay.append(row['ay'])
        az.append(row['az'])
        gx.append(row['gx'] / 57.2958)
        gy.append(row['gy'] / 57.2958)
        gz.append(row['gz'] / 57.2958)
        mx.append(row['mx'])
        my.append(row['my'])
        mz.append(row['mz'])

    next_time = 0
    with tqdm(total=df.shape[0]) as pbar:
        for index, r in df.iterrows():
            row_time = r['arduino_time']
            if next_time == 0:
                add(r)
                next_time = row_time + 20
            elif row_time >= next_time:
                add(r)
                next_time += 20
            pbar.update(1)

    new_df = pd.DataFrame(list(zip(imu, server_time, arduino_time, ax, ay, az, gx, gy, gz, mx, my, mz)),
                          columns=['imu', 'server_time', 'Timestamp[nanosec]', ' ax[m/s^2]', ' ay[m/s^2]', ' az[m/s^2]',
                                   ' gx[rad/s]', ' gy[rad/s]', ' gz[rad/s]', 'mx', 'my', 'mz'])
    return new_df


def phone_preprocess(df):
    with tqdm(total=df.shape[0]) as pbar:
        for index, r in df.iterrows():
            # df.at[index, 'Timestamp[nanosec]'] = r['Timestamp[nanosec]'] / 1000000
            pbar.update(1)
    return df


def get_files(dir):
    all_file_names = next(walk(dir), (None, None, []))[2]
    return all_file_names


def sync_data():
    phone_files = get_files('./phone/')
    imu_files = get_files('./imu/raw/')

    params = []

    for phone_file in phone_files:
        df_phone = pd.read_csv(f"phone/{phone_file}")
        df_phone = phone_preprocess(df_phone)
        for imu_file in imu_files:
            df_imu = pd.read_csv(f"./imu/raw/{imu_file}")
            df_imu = imu_preprocess(df_imu)
            df_imu.to_csv(f"./imu/preprocessed/{imu_file}", index=False)

            sync_indexes = sync_and_toGSC(df_phone, df_imu)
            sync_params = {'phone_file': phone_file, 'imu_file': imu_file, 'index_sync': sync_indexes}
            params.append(sync_params)

    with open("sync_params.json", "w") as outfile:
        json.dump(params, outfile)


def save_preview_image():
    with open("sync_params.json") as json_file:
        params = json.load(json_file)
        for sync_params in params:
            phone_file = sync_params['phone_file']
            imu_file = sync_params['imu_file']
            show_data(f"phone/{phone_file}", f"./imu/preprocessed/{imu_file}", sync_params, True)
            show_data(f"phone/{phone_file}", f"./imu/preprocessed/{imu_file}", sync_params, False)


if __name__ == '__main__':
    # sync_data()
    save_preview_image()
