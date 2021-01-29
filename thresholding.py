



import pandas as pd
from datetime import date
import os
import glob


today = date.today()
f_name = 'results_' + str(today)
results_path = '/home/vojtech/Nextcloud/python/UHFO_Pavel/results_generic/' + f_name + '/'
if not os.path.exists(results_path):
    os.mkdir(results_path)


env_path = '/media/vojtech/DATADRIVE4/hfo_envelopes/'

files = glob.glob(env_path + '/*/*/*.h5')


file = files[3]



def thresh_file(file):




