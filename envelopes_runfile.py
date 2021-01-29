#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:47:07 2020

@author: vojtech
"""



import sys, getopt
sys.path.append("/home/vojtech/Nextcloud/air-tools/")
sys.path.append("/home/vojtech/Nextcloud/Libraries_shared/")
import envelopes_single

def main(argv):
    
   results_path = '/media/vojtech/DATADRIVE4/hfo_envelopes/'

 
# =============================================================================
#    print('Number of arguments:', len(argv), 'arguments.')
#    print('Argument List:', str(argv)) 
# =============================================================================
   inputfile = ''
   outputfile = ''
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
       
         inputfile = arg
      
   envelopes_single.compute_envelopes(inputfile, results_path + inputfile.split('/')[-2] + '/low_var/', montage = 'low_var',  name_appendix = '_low_var_' + results_path.split('/')[-2],  freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000]])
   envelopes_single.compute_envelopes(inputfile, results_path + inputfile.split('/')[-2] + '/bipolar/', montage = 'bipolar',  name_appendix = '_bipolar_' + results_path.split('/')[-2],  freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000]])
   envelopes_single.compute_envelopes(inputfile, results_path + inputfile.split('/')[-2] + '/default/', montage = 'default',  name_appendix = '_default_' + results_path.split('/')[-2],  freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000]])



if __name__ == "__main__":
      main(sys.argv[1:])
    