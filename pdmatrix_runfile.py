


import sys, getopt
sys.path.append("/home/vojtech/Nextcloud/air-tools/")
sys.path.append("/home/vojtech/Nextcloud/Libraries_shared/")

import scipy
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
import glob
from multiprocessing import Pool
import time

from PDmatrixFunction import pd_subplot, pd_plot


def evaluate_folder(folder):
    
    files = glob.glob(folder + '/*.h5')
    for file in files:
        pdmatrix, table_pd = pd_plot(file, clear_path = '/'.join(file.split('/')[:-1]) + '/pd_matrix/',
                                 art_path = '/'.join(file.split('/')[:-1]) + '/pd_matrix_art/' )


def main(argv):
    
   t0 = time.time()
  
  

 
# =============================================================================
#    print('Number of arguments:', len(argv), 'arguments.')
#    print('Argument List:', str(argv)) 
# =============================================================================
   inputfolder = ''
   try:
      opts, args = getopt.getopt(argv,"hf:",["ifile="])
   except getopt.GetoptError:
      print('envelopes_runfile.py -f <inputfile>')
      sys.exit(2)
   for opt, arg in opts:
   
      if opt == '-h':
         print('envelopes_runfile.py -f <inputfile>')
         sys.exit()
      elif opt in ("-f", "--ifile"):
       
         inputfolder = arg
       
         
   folders = [ inputfolder + 'low_var', inputfolder + 'default', inputfolder + 'bipolar']    
        
   mp = Pool(3)
    
   try:
       results = mp.map(evaluate_folder, folders)
   except Exception as e:
       raise e    # I don't expect this to ever happen.
   finally:
       mp.terminate()
       
   t1 = time.time()
   total = t1 - t0
   print("PD matricies computation took %d seconds."  % total)


if __name__ == "__main__":
     main(sys.argv[1:])
    
  

   
    
    















