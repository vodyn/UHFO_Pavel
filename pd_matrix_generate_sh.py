#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:13:32 2020

@author: vojtech
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:38:03 2020

@author: vojtech
"""

import pandas as pd
import numpy as np
import glob

if __name__ == "__main__":
    
   
    
    path = '/media/vojtech/DATADRIVE4/hfo_envelopes/'


    folders = glob.glob(path + '/*/')
    
    
    
    base_command = 'ts /home/vojtech/pyenv/bin/python /home/vojtech/Nextcloud/python/UHFO_Pavel/scripts/'
      
       
    
    file = open("compute_pd_matrix.sh", "w") 
    
    file.write('#!/bin/bash\n')
     
    for i, folder in enumerate(folders):
        python_command = "pdmatrix_runfile.py -f "+ folder +"\n"
        file.write(base_command + python_command)
    
    
    file.close() 