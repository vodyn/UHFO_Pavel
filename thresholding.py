
import pandas as pd
from datetime import date
import os
import glob
from sphdf import read_sp_hdf
import numpy as np


today = date.today()
f_name = 'results_' + str(today)
results_path = '/home/vojtech/Nextcloud/python/UHFO_Pavel/results_generic/' + f_name + '/'
if not os.path.exists(results_path):
    os.mkdir(results_path)




def thresh_channel(ch_name, sig, std_multipl, fs):

    std_dev = np.std(sig)
    indexes = sig > (std_multipl *std_dev)

    det_sig = np.zeros(len(sig))
    det_sig[indexes] = 1
    starts = np.where(det_sig - np.append(det_sig[1:], det_sig[-1]) == -1)[0]
    stops = np.where(det_sig - np.append(det_sig[1:], det_sig[-1]) == 1)[0]

    # if det_sig starts with 1, start at the beginning has to be added
    if det_sig[0] == 1:
        starts = np.append(0, starts)
    if det_sig[-1] == 1:
        stops = np.append(stops, len(det_sig ) -1)

    # make sure that starts and stops have the same length
    if len(stops ) >len(starts):
         stops = stops[:-1]
    elif len(starts ) >len(stops):
         starts = starts[1:]

    starts = starts /fs *1e6
    stops = stops /fs *1e6

    # create detection df
    res = pd.DataFrame(data = {'channel': [ch_name ] *len(starts), 'start_time': starts, 'stop_time': stops,
                                'multipl': [std_multipl ] *len(starts)})

    return res


def thresh_file(file, std_multipl=13):

    print("Computing file " + file.split('/')[-1])

    # create results dataframe
    res = pd.DataFrame()

    data, info, fs = read_sp_hdf(file)
    ch_names = [info[index][0].decode() for index in np.arange(len(info))]

    for ch_name in ch_names:

        sig = data[ch_names.index(ch_name) ,:]
        res_appendix = thresh_channel(ch_name, data[ch_names.index(ch_name), :], std_multipl, fs)
        res = res.append(res_appendix)

    res['file'] = [file.split('/')[-1] ] *len(res)

    return res


def det_files(files, std_multipl=13):

    res_df = pd.DataFrame()
    for file in files:

        file_res = thresh_file(file, std_multipl=std_multipl)
        res_df = res_df.append(file_res)

    return res_df
