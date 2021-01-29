#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:53:26 2020

@author: vojtech
"""

import envelopes_single
import glob
import os
import numpy as np
import pandas as pd

data_path = '/media/vojtech/DATADRIVE2/data/'
results_path = '/media/vojtech/DATADRIVE4/hfo_envelopes/'


rest_files = pd.read_pickle('/media/vojtech/DATADRIVE2/data/rest_files.pkl')
rest_files = [data_path + '/'.join(file.split('/')[-2:]) for file in list(rest_files.session_path)]


envelopes_single.compute_envelopes(rest_files[6], results_path + 'low_var/', montage = 'low_var',  name_appendix = '_low_var_' + rest_files[6].split('/')[-2],  freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000]])


for i in np.arange(len(rest_files))[1:]:
    
    envelopes_single.compute_envelopes(rest_files[i], results_path + 'low_var/', montage = 'low_var',  name_appendix = '_low_var_' + rest_files[i].split('/')[-2],  freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000]])
    envelopes_single.compute_envelopes(rest_files[i], results_path + 'bipolar/', montage = 'bipolar',  name_appendix = '_bipolar_' + rest_files[i].split('/')[-2],  freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000]])
    envelopes_single.compute_envelopes(rest_files[i], results_path + 'default/', montage = 'default',  name_appendix = '_' + rest_files[i].split('/')[-2],  freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000]])
