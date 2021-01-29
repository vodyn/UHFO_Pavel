import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import matplotlib.pyplot as plt
from sphdf import read_sp_hdf
import time
import os
from numpy.ma import masked_array
from help_functions import split_electrode
import copy


def channel_delete(data):
    mean_data = np.mean(data)
    data[data < mean_data] = mean_data
    a = (np.array(data).std(axis=1))
    b = (np.array(data).sum(axis=1))
    b_std = np.std(b)
    b_mean = np.mean(b)
    b_std_p = b_mean + 3 * b_std

    # plt.figure()
    # plt.plot(a)
    # plt.axhline(0.005, color='g', linestyle='--', label='std')
    # plt.title('The std in each channel')
    # plt.xlabel('Number of channel')
    # plt.ylabel('Std')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(b)
    # plt.axhline(b_std_p, color='r', label='trashhold')
    # plt.axhline(np.mean(b), color='k', label='mean')
    # plt.title('The mean in each channel')
    # plt.xlabel('Number of channel')
    # plt.ylabel('Mean')
    # plt.show()

    dele = []
    for loc, val in enumerate(a):
        if val <= 0.005:
            dele.append(loc)
    for loc, val in enumerate(b):
        if val >= b_std_p:
            dele.append(loc)

    return dele


def pd_subplot(path, names):
    fig = plt.figure()
    for num, name in enumerate(names):
        path_file = path + name + ".h5"
        data, info, fs = read_sp_hdf(path_file)
        ch_names = [info[index][0].decode() for index in np.arange(len(info))]

        not_needed = ['EAS', 'fd1', 'mic', 'Fz', 'Cz', 'Pz',
                      'EOG1', 'EOG2', 'EOG3', 'EOG4', 'MS1', 'MS2', 'eA12', 'EKG2', 'EKG1', 'EKG4',
                      'CAL', 'SDT', 'EVT', 'BmRf', 'EKG3', 'ECG', 'Oz', 'EKG', 'FPz', 'FP1', 'FP2',
                      'FP1-Cz_f', 'FP2-Cz_f', 'Fz-Cz_f', 'Cz-Oz_f', 'Scalp']
        dele_ch = []
        for p, q in enumerate(ch_names):
            for o in not_needed:
                if o in q:
                    dele_ch.append(p)

        data = np.delete(data, dele_ch, axis=0)
        ch_names = np.delete(ch_names, dele_ch, axis=0)

        [pdmatrix, vmin, vmax, time_s, position_s, dele] = pd_draw(data, fs, name, ch_names, num_pixel=1800)

        # Pouze na odladění skriptu
        print(time_s)
        print(position_s)
        print(dele)

        if num <= 3:
            ax1 = plt.subplot2grid((4, 4), (0, num), fig=fig)
            sns.heatmap(pdmatrix, vmin=vmin, vmax=vmax, cmap='Greys', square=False, cbar=False, ax=ax1)
        elif 3 < num <= 7:
            ax2 = plt.subplot2grid((4, 4), (1, num - 4), fig=fig)
            sns.heatmap(pdmatrix, vmin=vmin, vmax=vmax, cmap='Greys', square=False, cbar=False, ax=ax2)
        elif 7 < num <= 11:
            ax3 = plt.subplot2grid((4, 4), (2, num - 8), fig=fig)
            sns.heatmap(pdmatrix, vmin=vmin, vmax=vmax, cmap='Greys', square=False, cbar=False, ax=ax3)
        else:
            ax4 = plt.subplot2grid((4, 4), (3, num - 12), fig=fig)
            sns.heatmap(pdmatrix, vmin=vmin, vmax=vmax, cmap='Greys', square=False, cbar=False, ax=ax4)

    plt.show()


def pd_plot(path, clear_path = None, art_path = None, num_pixel=2560):
    
    
    
    global start
    t0 = time.time()
    data, info, fs = read_sp_hdf(path)
    ch_names = [info[index][0].decode() for index in np.arange(len(info))]
    t1 = time.time()
    total = t1 - t0
    print("Data loading takes %d seconds."
          % total)
    name = path.split('/')[-1].split('.')[0]

    row_pd, col_pd = data.shape

    # Vymazání nepotřebných kanálů
    not_needed = ['EAS', 'fd1', 'mic', 'Fz', 'Cz', 'Pz',
                  'EOG1', 'EOG2', 'EOG3', 'EOG4', 'MS1', 'MS2', 'eA12', 'EKG2', 'EKG1', 'EKG4',
                  'CAL', 'SDT', 'EVT', 'BmRf', 'EKG3', 'ECG', 'Oz', 'EKG', 'FPz', 'FP1', 'FP2',
                  'FP1-Cz_f', 'FP2-Cz_f', 'Fz-Cz_f', 'Cz-Oz_f', 'Scalp']
    dele_ch = []
    for p, q in enumerate(ch_names):
        for o in not_needed:
            if o in q:
                dele_ch.append(p)

    data = np.delete(data, dele_ch, axis=0)
    ch_names = np.delete(ch_names, dele_ch, axis=0)
    
    #reorder data
    ch_names_split = [split_electrode(x) for x in ch_names]
    
    names_df = pd.DataFrame(data = {'names':ch_names})
    names_df['electrodes'] = list(list(zip(*ch_names_split))[0])
    names_df['numbers'] =  list(list(zip(*ch_names_split))[1])
    order = list(names_df.sort_values(by=['electrodes','numbers']).index)
    ch_names = [ch_names[i] for i in order]
    data_new = [data[i,:] for i in order]
    data = np.array(data_new)
    
    
    # Funkce na vytvoření PDmatice
    [pdmatrix, time_s, position_s] = pd_draw_v(data, fs, num_pixel)

    # Vykreslení PD matice
    cc = plt.figure()
    cc.set_size_inches((25.98, 14.56), forward=False)
    plt.imshow(pdmatrix, interpolation='None', vmin=0, cmap=plt.cm.Greys, aspect="auto")
    plt.xticks(position_s, time_s)
    row, col = pdmatrix.shape
    plt.yticks(range(0, row), ch_names,  fontsize= (num_pixel/16*9*0.7)//len(ch_names))
    plt.tick_params(labelright=True,right = True, labeltop = True, top = True)
    plt.xlabel('Time [sec]', fontsize=15)
    plt.ylabel('Channel', fontsize=15)
    plt.title(path.split('/')[5] + '    ' + name, fontsize=20)
    
    if isinstance(clear_path, str):
             
        if not os.path.exists(clear_path):

            os.mkdir(clear_path) 
                          
        cc.savefig(clear_path + path.split('/')[-1].split('.')[0] + '.png',
                   bbox_inches='tight',pad_inches=0.2, dpi = 300)
            
  
    # funkce pro odstranění poškozených kanálů
    dele = channel_delete(data)
    print("Noisy channels in file", name, ":", np.take(ch_names, dele))
    # Funkce na vyhledání artefaktů v PDmatici
    pdmatrix_art, peaks, art_min, art_max = artifacts_in_pdmatrix(pdmatrix, dele)

    
    num_sec = int(col_pd / fs)
    time_pixel = num_sec / len(pdmatrix[0])

    table_pd = pd.DataFrame(zip(art_min*time_pixel,art_max*time_pixel))
    table_pd = table_pd.rename(columns = {0:'start_time', 1:'stop_time'})
    table_pd['file'] = [name]*len(table_pd)
    table_pd['art_type'] = ['pd_matrix']*len(table_pd)
    
    channel_df = pd.DataFrame(data = {'channel': np.take(ch_names, dele), 'file':[name]*len(dele)})

    # Vykreslení PD matice s červeně označenými artefakty
    cc = plt.figure()
    cc.set_size_inches((25.98, 14.56), forward=False)
    pdmatrix_a = masked_array(pdmatrix_art, pdmatrix_art < 90)
    pdmatrix_b = masked_array(pdmatrix_art, pdmatrix_art >= 90)
    plt.imshow(pdmatrix_a, interpolation='None', vmin=99, vmax=100.5, cmap=plt.cm.Reds, aspect="auto")
    plt.imshow(pdmatrix_b, interpolation='None',  vmin=0, cmap=plt.cm.Greys, aspect="auto")
    plt.xticks(position_s, time_s)
    row, col = pdmatrix.shape
    plt.yticks(range(0, row), ch_names,  fontsize= (num_pixel/16*9*0.7)//len(ch_names))
    plt.tick_params(labelright=True,right = True, labeltop = True, top = True)
    plt.xlabel('Time [sec]', fontsize=15)
    plt.ylabel('Channel', fontsize=15)
    plt.title(path.split('/')[5] + '    ' + name, fontsize=20)
    
    if isinstance(art_path, str):
             
        if not os.path.exists(art_path):

            os.mkdir(art_path) 
                          
        cc.savefig(art_path + path.split('/')[-1].split('.')[0] + '.png',
                   bbox_inches='tight', pad_inches=0.2, dpi = 300)
        
        time_path = art_path + art_path.split('/')[-4] + '_' + art_path.split('/')[-3] + '_time.pkl'
        el_path = art_path + art_path.split('/')[-4] + '_' + art_path.split('/')[-3] + '_el.pkl'
        
        if os.path.isfile(time_path):
            cc = pd.read_pickle(time_path)
            table_pd = cc.append(table_pd)   
            table_pd = table_pd.drop_duplicates().reset_index(drop=True)
        
        table_pd.to_pickle(time_path)
        
        if os.path.isfile(el_path):
            cc = pd.read_pickle(el_path)
            channel_df = cc.append(channel_df)  
            channel_df = channel_df.drop_duplicates().reset_index(drop=True)
        
        channel_df.to_pickle(el_path)
    
    return pdmatrix, table_pd


def pd_draw(data, fs, name, ch_names, num_pixel=2560):

   

    ch_means = np.array(data).mean(axis=1)
    for n, ch in enumerate(ch_means):
        data[n][np.where(data[n] < ch)] = 0

    # Normalizace dat
    max_data = np.max(data)
    min_data = np.min(data)
    data_norm = (data - min_data) / (max_data - min_data)
    
    #Zjisteni delky v s
    len_s = np.shape(data[1])[0]/fs
        
    # Parametry ndarray
    row, col = data_norm.shape
    window = col // num_pixel

    # Podvzorkování pd matice
    pdmatrix = np.zeros(shape=(row, num_pixel))
    for number, data_for in enumerate(data_norm):
        for ind in range(0, num_pixel):
            pdmatrix[number][ind] = np.max(data_for[((window * ind) + 1): (window * (ind + 1))])

    # Nastavení časové osy
  
    time_s = np.arange(0,len_s,200).astype(int)
    position_s = (time_s*(np.shape(pdmatrix[1])[0]/len_s)).astype(int)

    # Data oříznutá o extrémy
    pdmatrix_a = pdmatrix
    per = np.percentile(pdmatrix, 95)
    pdmatrix[pdmatrix >= per] = per

    # Nastavení vmin a vmax
    vmin = np.mean(pdmatrix)
    vmax = np.mean(pdmatrix) + 2 * np.std(pdmatrix)
    pdmatrix[pdmatrix >= vmax] = vmax

    return pdmatrix_a, vmin, vmax, time_s, position_s, dele

def pd_draw_v(data, fs, num_pixel=2560):

    sens = 10

    ch_means = np.array(data).mean(axis=1)
    ch_maxs = data.max(axis=1)
    multipl = np.max((ch_maxs-ch_means)/ch_means)
    
    data_norm = (data.T - ch_means).T
     
    data_norm = sens * (data_norm/multipl)
     

    
    #Zjisteni delky v s
    len_s = np.shape(data[1])[0]/fs
        
    # Parametry ndarray
    row, col = data_norm.shape
    window = col // num_pixel

    # Podvzorkování pd matice
    pdmatrix = np.zeros(shape=(row, num_pixel))
    for number, data_for in enumerate(data_norm):
        for ind in range(0, num_pixel):
            pdmatrix[number][ind] = np.max(data_for[((window * ind) + 1): (window * (ind + 1))])

    # Nastavení časové osy
  
    time_s = np.arange(0,len_s,100).astype(int)
    position_s = (time_s*(np.shape(pdmatrix[1])[0]/len_s)).astype(int)

    # Data oříznutá o extrémy
    per = 98
    target = 0.19
    m=0
    i=0
    pdmatrix_base = copy.deepcopy(pdmatrix)
    while (m<target-0.01) | (m>target+0.01):
        pdmatrix = copy.deepcopy(pdmatrix_base)
        per_v = np.percentile(pdmatrix, per)
        pdmatrix[pdmatrix >= per_v] = per_v
        pdmatrix = pdmatrix/per_v
        m = np.mean(pdmatrix)
        per = per-(target-m)*5
        if per>100:
            per = 99.9
        elif per<=0:
            per = 0
        i+=1
        if i>15:
            break


    return pdmatrix, time_s, position_s


def artifacts_in_pdmatrix(pdmatrix, dele):
    x = (np.array(pdmatrix).mean(axis=0)) ** 2
    x[x < np.mean(x) + 2 * np.std(x)] = 0

    peaks, properties = signal.find_peaks(x, height=np.mean(x) + 2 * np.std(x), prominence=0)

    # Zobrazení postupu určování pozic artefaktů

    # plt.figure()
    # plt.subplot(211)
    # plt.axhline(np.mean(x) + 2.5 * np.std(x), color='r', linestyle='--', label='mean')
    # plt.plot(x)
    # plt.xlim(0, 1800)
    # plt.plot(peaks, x[peaks], marker='x', color='g', markersize=10)
    # plt.axhline(np.mean(x), color='k', linestyle='--', label='mean')
    # plt.axhline(np.mean(x) + 2 * np.std(x), color='r', label='treshold')
    # plt.subplot(212)
    # sns.heatmap(pdmatrix, vmin=vmin, vmax=vmax, cmap='Greys', square=False, cbar=False)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(x)
    # plt.plot(peaks, x[peaks], "x")
    # plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"], ymax=x[peaks], color="C1")
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(211)
    # z = (np.array(pdmatrix).mean(axis=0))
    # z[z < np.mean(z) + 2 * np.std(z)] = 0
    # plt.plot(z)
    # plt.subplot(212)
    # y = (np.array(pdmatrix).std(axis=0))
    # tui = y - np.mean(y)
    # tui = signal.medfilt(tui, 3)
    # plt.plot(tui)
    # plt.show()

    # Dopočítání pouzic artefaktů v krajích pdmatic
    art_min = properties['left_bases']
    art_max = properties['right_bases']

    if x[0] >= np.mean(x) + 2 * np.std(x):
        art_min = np.append(art_min, 0)
        art_max = np.append(art_max, 3)
    if x[len(x) - 1] >= np.mean(x) + 2 * np.std(x):
        art_min = np.append(art_min, len(x)-3)
        art_max = np.append(art_max, len(x))

    # artefakty odděleny pomocí odlišné hodnoty
    for index in range(0, len(art_min)):
        pdmatrix[:, art_min[index] + 1:art_max[index]] = 100

    pdmatrix[dele] = 100.1
    pdmatrix_art = pdmatrix

    return pdmatrix_art, peaks, art_min, art_max
