
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


env_path = '/media/vojtech/DATADRIVE4/hfo_envelopes/'
data_path = '/home/vojtech/Nextcloud/python/UHFO_Pavel/results_generic/results_2021-02-02/thresh_df.pkl'

files = glob.glob(env_path + '/*/*/*.h5')
files_temp = [file.split('/')[-1] for file in files]



data = pd.read_pickle(data_path)

#prepare files df
file_df = pd.DataFrame(data = {'files':data.file.unique()})
file_df['high_env_freq'] = file_df.files.apply(lambda x: x.split('env')[2].split('.')[0])
file_df['montage'] = file_df.files.apply(lambda x: x.split('_')[-4])
file_df.loc[file_df.montage =='var', 'montage'] = 'low_var'

stat_df = pd.DataFrame()
for x, file_row in file_df.iterrows():

    # find file location
    file_loc = files[files_temp.index(file_row.files)]
    # read file info
    data_row, info, fs = read_sp_hdf(file_loc)
    ch_names = [info[index][0].decode() for index in np.arange(len(info))]

    # file length in microseconds
    file_len = len(data_row[0,:])/fs*1e6
    # load thresholded data
    data_temp = data[data.file==file_row.files]
    data_temp = data_temp.reset_index(drop=True)

    for multipl in data_temp.multipl.unique():


        #prepare file results df
        res = pd.DataFrame(data = {'file': [file_row.files]*len(ch_names),
                                   'high_env_freq':[file_row.high_env_freq]*len(ch_names),
                                   'montage': [file_row.montage]*len(ch_names), 'multipl': [multipl]*len(ch_names),
                                   'channel': ch_names, 'ratio[%]':[0]*len(ch_names)})


        data_temp['leng'] = data_temp.stop_time - data_temp.start_time
        for channel in res.channel.unique():
            sum_leng = np.sum(data_temp.loc[data_temp.channel == channel, 'leng'])
            if sum_leng > 0:
                res.loc[res.channel == channel, 'ratio[%]'] = sum_leng/file_len*100     #value in percent



        print(file_row.files)
        stat_df = stat_df.append(res)