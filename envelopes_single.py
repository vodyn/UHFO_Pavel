#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:55:50 2019

@author: vojtech
"""

import pydread
import numpy as np
import cupy as cp
import h5py
import time
import glob
from pymef import mef_session
from sphdf import read_sp_hdf, write_sp_hdf
import os
import re
from pymef.mef_session import MefSession
import pandas as pd


from help_functions import remove_utility_channels, get_electrode_type
from montage import bipolar_montage, low_var_montage
'''
Spike (S): 0-80 Hz, lze použít spodní filtr do cca 5 Hz
Ripples (R): 80-200 Hz
Fast Ripples (FR): 200-500 Hz
Very Fast Ripples (VFR): 500-1000 Hz
Ultra Fast Ripples (UFR): 1000-2000 Hz
'''

    


# =============================================================================
# #FNUSA
# dest_path = '/media/vojtech/0a1c3f84-2131-4d9e-be62-f1be739347a7/397/'
# raw_file ='/media/vojtech/0a1c3f84-2131-4d9e-be62-f1be739347a7/397/MSEL_00397_5000.h5'
# 
# =============================================================================
#ISIBrno
raw_file = '/media/vojtech/DATADRIVE3/FNUSA_high_fs/seeg-061-25kHz/Easrec_hdeeg25k-seeg061-pa2_191029-1315.d'
dest_path = '/media/vojtech/DATADRIVE3/envelopes/61_25kHz_1/'

# =============================================================================
# #Doma
# dest_path = '/home/vojtech/Nextcloud/python/UHFO/test_data/'
# raw_file = '/home/vojtech/Nextcloud/python/UHFO/test_data/test_data.h5'
# =============================================================================


def get_channel_info(channel_name):
    """
    Desides what cathegory the channel belongs to (Macro,Micro,Util,Scalp)
    
    Parameters:
    -----------
    channel_name - name of the channel\n
    """

    utility_channels = ['A12A','A12B','A12C','A12D','AU12',                
                        'EKG','EKG1','EKG2','EKG3','EKG4',
                        'EMG1','EMG2',
                        'EOG1','EOG2','EOG3','EOG4',
                        'EVT','SDT','CAL','EEG2','EAS',
                        'foto','mic','eA12','fd1']

    scalp_channels = ['Cz','Fz','MS1','MS2','Ms1','Ms2','Pz','Oz']

    

    if channel_name in scalp_channels:
        channel_info = 'scalp'
    elif channel_name in utility_channels:
        channel_info = 'util'
    elif 'm' in channel_name:
        if 'Rf' in channel_name:
            channel_info = 'util'
        else:
            channel_info = 'micro'
    else:
        channel_info = 'macro'
        
    return channel_info

def save2mef3(mef_session_path, sh, xh, data, recording_type='relax',
           slice_len=600, pass_1='tech', pass_2='bemena',
           samps_per_mef_block=5000):
      
   
    """
    Function to convert .d file into a mef3 session
    
    Parameters
    ----------
    file_path: str
        Full path to .d file
    mef_session_path: str
        Path to where the mef session is going to be written (indluding .mefd
        directory)
    recording_type: str
        Type of the recording beeing converted (rest, oddball, etc)
    slice_len: int
        Length of the data slice in seconds
    pass_1: str
        Level 1 password
    pass_2: str
        Level 2 password
    samps_per_mef_block: int
        Samples per mef block
    """


     
    if not os.path.exists(mef_session_path):
       os.mkdir(mef_session_path)
    
    
    # Check if d-file is valid
    #if sh['data_info']['data_invalid']:
    #    return (d_file_path+'; invalid data')
        
    if sh['unit']:
        unit_conversion = 1/sh['unit']
    else:
    #    return str(d_file_path+'; unit conversion factor is zero.')
        unit_conversion = None
        


    
    section2_arr = np.zeros(1, mef_session.create_tmd2_dtype())
    section3_arr = np.zeros(1, mef_session.create_md3_dtype())
    
    section3_arr['GMT_offset'] = 3600
    section3_arr['recording_location'] = 'FNUSA'
    section3_arr['recording_time_offset'] = int(xh['time_info'] * 1e6)
    
    section2_arr['recording_duration'] = int((sh['nsamp'] / sh['fsamp'])*1e6)
    section2_arr['reference_description'] = 'intracranial_average'
    section2_arr['sampling_frequency'] = sh['fsamp']/5
    section2_arr['AC_line_frequency'] = 50
    section2_arr['units_description'] = 'uV'
    section2_arr['start_sample'] = 0
    
    if recording_type:
        section2_arr['session_description'] = recording_type
        #section2_dict['session_description'] = recording_type
        
    if unit_conversion:
        section2_arr['units_conversion_factor'] = unit_conversion
        #section2_dict['units_conversion_factor'] = unit_conversion
    
    # Open mef session
    os.makedirs(mef_session_path, exist_ok=True)
    #ms = MefSession(mef3_path+d_folder+'/'+session_folder, 'bemena', False, True)
    

    done_channels = []
    for ci,channel_name in enumerate(xh['channel_names']):   
        
        # Check for doubled channels
        if channel_name in done_channels:
            new_chan_i = 1
            while True:
                new_chan_name = channel_name+'_'+str(new_chan_i)
                if new_chan_name in done_channels:
                    new_chan_i += 1
                else:
                    break
                
            print('--------- Duplicate channel '+channel_name+' renaming to '+new_chan_name)
            
            channel_name = new_chan_name
            
        done_channels.append(channel_name)
        
        channel_folder = channel_name+'.timd'
               
        # We will have only one segment in this case
        segment_folder = channel_name+'-000000.segd'
        segment_path = mef_session_path+'/'+channel_folder+'/'+segment_folder+'/' 
        
        os.makedirs(segment_path, exist_ok=True)
        
        # Modify metadata
        section2_arr['channel_description'] = get_channel_info(channel_name)
        section2_arr['acquisition_channel_number'] = ci
        
        # Write the metadata file
        mef_session.write_mef_ts_metadata(segment_path,
                                     pass_1,
                                     pass_2,
                                     int(xh['time_info'] * 1e6),
                                     int(xh['time_info'] * 1e6 + ((sh['nsamp'] / sh['fsamp']) * 1e6)),
                                     section2_arr,
                                     section3_arr)
        
        # Write the data file
        mef_session.write_mef_ts_data_and_indices(segment_path,
                                             pass_1,
                                             pass_2,
                                             samps_per_mef_block,
                                             data[ci].astype('int32'),
                                             0)
        

        
         
    # Take care of records
    print('Writing records')
    record_list = []
    record_file_path = mef_session_path+'/'
    print(record_file_path)

    if len(record_list):
        mef_session.write_mef_data_records(record_file_path,
                                      pass_1,
                                      pass_2,
                                      int(xh['time_info'] * 1e6),
                                      int(xh['time_info'] * 1e6 + ((sh['nsamp'] / sh['fsamp']) * 1e6)),
                                      int(xh['time_info'] * 1e6),
                                      record_list)
    return 0
        

def envelope(inputs):

    '''
    y: fft of signal
    fs: sampling frequency
    fmin: low cutoff frequency
    fmax: high cutoff frequency
    '''
    
# =============================================================================
#     #clear gpu memory
#     memory=cp.get_default_memory_pool()
#     memory.free_all_blocks()
# =============================================================================
    
    y, fs, fmin, fmax = inputs

    N = len(y)

    if fmin > 0:
        imin = int(fmin//(fs/N))
    else:
        imin = int(0)
        y[0] = y[0]/2    # ss se neopakuje, nutno elimin. nas. 2 na konci

    if fmax < fs/2:
        imax = int(fmax//(fs/N))
    else:
        imax = int(N//2)

    yy = np.zeros(len(y), dtype=np.complex64)
    istred = (imax+imin)//2
    ld = imax - istred
    yy[0:ld] = y[istred:imax]
    lh = istred - imin
    yy[-lh::] = y[imin:istred]
       
    YY = cp.asnumpy(cp.abs(cp.fft.ifft(cp.array(yy)))*2)
    YY = YY*YY
    YY=np.convolve(YY, np.ones((8,))/8, mode='same')
    YY = YY.astype('float32')
    
    #get power envelope

    return YY[4::5]

def compute_fft(signal):
    
# =============================================================================
#     #clear gpu memory
#     memory=cp.get_default_memory_pool()
#     memory.free_all_blocks()
# =============================================================================
    
    y = cp.asnumpy(cp.fft.fft(cp.array(signal)))
    return y

# Data loading
# =============================================================================
# raw_file = '/media/vojtech/0a1c3f84-2131-4d9e-be62-f1be739347a7/Jozka_data/' \
#             '72/Easrec_sciexp-seeg072_180314-1245.d'
# #     
#  '/media/mdpm/d04/eeg_data/kuna_eeg/data-d_fnusa_organizace/seeg/' \
#  'seeg-061-170531/Easrec_sciexp-seeg061_170531-1038.d'
#  
#  '/media/mdpm/d04/eeg_data/kuna_eeg/data-d_fnusa_organizace/seeg/' \
#  'seeg-072-180314/Easrec_sciexp-seeg072_180314-1245.d'
#  
#  '/media/mdpm/d04/eeg_data/kuna_eeg/data-d_fnusa_organizace/seeg/' \
#  'seeg-079-180919/Easrec_sciexp-seeg079_180919-0847.d'
 #  '/media/mdpm/d04/eeg_data/kuna_eeg/data-d_fnusa_organizace/seeg/' \
#  'seeg-003-120118/rroman_based_1201181020.relax.d   
    
#    
#   'path = '/media/vojtech/DATADRIVE3/envelopes/61/'
# =============================================================================

def compute_envelopes(raw_file, dest_path, montage = None, start = None, stop = None, name_appendix =  '',file_type_out = 'hdf', password = 'bemena',  freq_bands = None, channels = None):
    '''
    raw file:
        full path to raw file
    dest path
        folder to save all envelopes in
    file type
        file type to save envelopes in 
        file_type = 'hdf'
        file_type = 'mef'
    '''
    
    
    #verify dest path
    if not os.path.exists(dest_path):
        if not os.path.exists('/'.join(dest_path.split('/')[:-2])):
            os.mkdir( '/'.join(dest_path.split('/')[:-2]))
                              
        os.mkdir(dest_path)
        
    
# =============================================================================
#     path = '/media/vojtech/DATADRIVE3/envelopes/72/'
#     file = 'Easrec_sciexp-seeg072_180314-1245.d'
#     dest_path = '/media/mdpm/d04/eeg_proc/vojta_proc/72/'
#     raw_file = path + file
# =============================================================================
    t_start = time.time()
    file_type_in = raw_file.split('.')[-1]
    file_name = raw_file.split('/')[-1]
    file_name = file_name.split('.')[-2] + name_appendix
    if file_type_in == 'd':
    
        sh,xh = pydread.read_d_header(raw_file)       
        fs = sh['fsamp']
        nsamp = sh['nsamp']
        t0 = time.time()
        ch_names = xh['channel_names']
        ch = list(np.arange(len(ch_names)))
        data = pydread.read_d_data(raw_file, ch, 0, nsamp)
    
        t1 = time.time()
        total = t1-t0
        print('Data loading takes %d seconds.' 
              % (total))
        
        
    elif file_type_in == 'h5':
        t0 = time.time()
        data, info, fs = read_sp_hdf(raw_file)
        data = data.T
        ch_names = [info[index][0].decode() for index in np.arange(len(info))]
        t1 = time.time()
        total = t1-t0
        print('Data loading takes %d seconds.' 
              % (total))
        
    elif file_type_in == 'mefd':
         ms = MefSession(raw_file, password)
         channel_info = ms.read_ts_channel_basic_info()
         if channels == None:
            
            ch_names = [channel_info[x]['name'] for x in np.arange(len(channel_info)) ]
            ch_names = remove_utility_channels(ch_names)
            
            channels = pd.DataFrame({'ch_names':ch_names})
            channels['electrode_type'] = channels.ch_names.apply(lambda x: get_electrode_type(x))
            ch_names = list(channels.ch_names)
            #ch_names = list(channels[channels.electrode_type=='micro'].ch_names)
         else:
            ch_names = channels
            
         
         sig = ms.read_ts_channels_uutc(ch_names, [start, stop])
         data = np.stack(sig).T
         ch_names_help = [channel_info[x]['name'] for x in np.arange(len(channel_info)) ]
         ind = ch_names_help.index(ch_names[0])
         fs = int(ms.read_ts_channel_basic_info()[ind]['fsamp'][0])
         ms.close()
    
    if freq_bands ==None:
        freq_bands = [[5, 80], [80, 200], [200, 500], [500, 1000], [1000, 2000],
                  [2000, 4000], [4000, 8000]]

    if montage == 'low_var':
        print('Computing low variance motage.')
        data = low_var_montage(data)
       
    elif montage =='bipolar':
        print('Computing bipolar montage.')
        data, ch_names = bipolar_montage(data, ch_names)
        
    max_freqs = [freq[1] for freq in freq_bands]
    max_band = np.max(np.cumsum(np.array(max_freqs)<fs/2))
    freq_bands = freq_bands[0:max_band]
        
    # Clear data
    t0 = time.time()
    not_needed = ['EAS', 'fd1', 'mic', 'Fz', 'Cz', 'Pz',
    'EOG1', 'EOG2', 'EOG3', 'EOG4', 'MS1', 'MS2', 'eA12', 'EKG2', 'EKG1', 'EKG4',
    'CAL', 'SDT', 'EVT', 'BmRf', 'EKG3']
        
    index=[]
    for i in np.arange(len(not_needed)):
        try:
            index.append(ch_names.index(not_needed[i]))
        except:
            print('Channel name ' + not_needed[i] + ' cannot be removed, ' \
                  'because is not in recording.' )
            
    data = np.delete(data, index, 1)
    for ind in sorted(index, reverse=True):
            del ch_names[ind]
    t1 = time.time()
    total = t1-t0
    print('Data clearence takes %d seconds.' %(total))
    
    # Compute fft
    shape = np.shape(data)
    t0 = time.time()
    fft = np.zeros(shape, dtype=np.complex64)
    for channel_num, channel in enumerate(data.T):
          fft[:, channel_num] = compute_fft(channel)
          print(' Computing fft of channel %d.' %(channel_num))
    t1 = time.time()
    total = t1-t0
    print('FFT for all channels on GPU without multiprocessing took %d seconds.' \
          %(total))
    
    # Memory maping
    t0 = time.time()
    f = np.memmap(dest_path + 'data_fft.dat', dtype=np.complex,
                  mode='w+', shape=shape)
    f[:] = fft[:]
    
    del fft
    del f
    f = np.memmap(dest_path + 'data_fft.dat', dtype=np.complex,
                  shape=shape)
    t1 = time.time()
    total = t1-t0
    print('Memory mapping took %d seconds.' %(total))
    

    
    xx = [(ch_name, 'RAW', 'uV') for ch_name in ch_names]
    sig_info = np.array(xx   ,dtype=[('ChannelName','S256'), ('DatacacheName','S256'), ('Units','S256')])
    
    for freq_num, freqs in enumerate(freq_bands):
        t0 = time.time()
        envelopes = np.zeros([shape[0]//5,shape[1]], dtype=np.float32)
        for channel in np.arange(shape[1]):
            inputs = [f[:,channel],fs, freqs[0], freqs[1]]
            envelopes[:, channel] = envelope(inputs)
            print(' Computing envelope of channel %d in band from %d to %d Hz' \
                  %(channel, freqs[0], freqs[1]))
        if file_type_out == 'hdf':
            print('Saving to hdf file.')
            write_sp_hdf(dest_path + file_name + '_env'+ str(max_freqs[freq_num]) + '.h5', envelopes.T, sig_info, fs//5)
# =============================================================================
#             with h5py.File(dest_path + file_name + '_env'+ str(max_freqs[freq_num]) + '.h5', 'w') as file:
#                 file.create_dataset('Data', data = envelopes.T)
#                 file.create_dataset('Info', data = sig_info) 
#                 file.attrs['Fs'] = fs/5
#                 file.close()
# =============================================================================
        elif file_type_out == 'mef3':
            file_path = dest_path + 'env'+ str(max_freqs[freq_num]) +'.mefd'
            print('Saving to mef3 file.')
            save2mef3( file_path, sh, xh, envelopes,
                      recording_type='power_envelope')
        else:
            print('Cannot save to ' + file_type_out + '.')
            break
            
        del envelopes    
        t1 = time.time()
        total = t1-t0
        print('Envelope in band from %d Hz to %d Hz on GPU without multiprocessing' \
               'took %d seconds.' %(freqs[0], freqs[1], total))
        
    t_stop = time.time()
    total_time = t_stop - t_start
    print('Computing all envelopes took %d seconds.' %(total_time))
    
    os.remove(dest_path + 'data_fft.dat')


# =============================================================================
# 
# 
# paths = ['/media/vojtech/DATADRIVE3/FNUSA_high_fs/seeg-061-25kHz/*.d', '/media/vojtech/DATADRIVE3/FNUSA_high_fs/seeg-072-25kHz/*.d', '/media/vojtech/DATADRIVE3/FNUSA_high_fs/seeg-079-25kHz/*.d']
# 
# for pat in paths:
# 
#     files = glob.glob(pat)
#     
#     
#     for file in files:
#         dest_path = '/media/vojtech/DATADRIVE2/envelopes/' +file.split('/')[-1][:-2] + '/'
#         if not os.path.exists(dest_path):
#             os.mkdir(dest_path)
#         compute_envelopes(file, dest_path, file_type_out = 'hdf')
# =============================================================================
