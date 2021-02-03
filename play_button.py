

import process_detections
import pandas as pd
import thresholding
import glob
from datetime import date
import os


today = date.today()
f_name = 'results_' + str(today)
results_path = '/home/vojtech/Nextcloud/python/UHFO_Pavel/results_generic/' + f_name + '/'
if not os.path.exists(results_path):
    os.mkdir(results_path)



env_path = '/media/vojtech/DATADRIVE4/hfo_envelopes/'
raw_files = glob.glob(env_path + '/*/*/*.h5')

thresh_folder = '/home/vojtech/Nextcloud/python/UHFO_Pavel/results_generic/thresh_13_data/'

def threshold_envelopes(files):
    chunk = 100
    i=0
    while (i+2)*chunk <= len(files):
        thresh_data = thresholding.det_files(files[i*chunk:(i+1)*100], std_multipl=13)
        thresh_data.to_pickle(results_path + 'thresh_data_' + str(i) + '.pkl')
        i+=1

    thresh_data = thresholding.det_files(files[i*chunk:len(files)], std_multipl=13)
    thresh_data.to_pickle(results_path + 'thresh_data_' + str(i) + '.pkl')


def process_thresh_data(thresh_folder, raw_files):

    files = glob.glob(thresh_folder + '*.pkl')
    res = pd.DataFrame()

    for file in files:
        res = res.append(process_detections.proces_thresh(file, raw_files))

    return res

processed_data = process_thresh_data(thresh_folder, raw_files)
processed_data = process_thresh_data.reset_index(drop=True)

processed_data.to_pickle(results_path + 'processed_data.pkl')