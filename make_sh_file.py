#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:38:03 2020

@author: vojtech
"""

import pandas as pd
import numpy as np

if __name__ == "__main__":
    
   
    
    
    data_path = '/media/vojtech/DATADRIVE2/data/'
    results_path = '/media/vojtech/DATADRIVE4/hfo_envelopes/'
    
    rest_files = pd.read_pickle('/media/vojtech/DATADRIVE2/data/rest_files.pkl')
    rest_files = [data_path + '/'.join(file.split('/')[-2:]) for file in list(rest_files.session_path)]
    
    
    base_command = 'ts /home/vojtech/pyenv/bin/python /home/vojtech/Nextcloud/python/UHFO_Pavel/scripts/'
  
       
    
    file = open("compute_envelopes.sh", "w") 
    
    file.write('#!/bin/bash\n')
 
    for i in np.arange(len(rest_files)):
        python_command = "envelopes_runfile.py -f "+ rest_files[i] +"\n"
        file.write(base_command + python_command)
    
    
    file.close() 

