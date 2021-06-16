from __future__ import print_function
import sys
import os
from pathlib import Path
import numpy as np
import glob
import pandas as pd
import seaborn as sns
from pandas import HDFStore
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['text.latex.unicode']=True
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#########################################
''' Reading and Plotting Sample Data'''
#########################################

current_direc = os.getcwd()
#print('Current Working Directory is : {} '.format(current_direc))
data_direc = os.listdir('.')                  ## This will read current directory in which we are working.(/current_directory)
#print(data_direc)
## This will find the files starts with digit 32 in the current working directory.
sample_data_direc_select = [x for x in data_direc if x.startswith('33')]
#print(len(sample_data_direc_select))
#print(sorted(sample_data_direc_select))

i = 1 ### injection signal value
for sample_data_direc in sorted(sample_data_direc_select):

    #print(sample_data_direc)                   ## This will give all data directories with in the current working directories
    current_direc_data_file = os.path.join(current_direc,
                                           sample_data_direc)           ## This  command will call the data directories.
    #print('Current Sample Data Direc is  {}'.format(current_direc_data_file))
    sample_data_file_name = os.path.join(current_direc_data_file,
                                         'sample_param_'+str(i)+'_result.h5')      ## This will call the data file with label 'label_result.h5'.
    #print('Sample Data File Name is  {}'.format(sample_data_file_name))

    ## Loading the datafile in reading mode.
    sample_data_file_open = pd.HDFStore(sample_data_file_name,'r')  ## This command will load the data file in read mode.
    #print('Sample Data file opened is {}'.format(sample_data_file_open))
    for keys in sample_data_file_open:  ## This command will read main keys and sub keys in the data file
        print(keys)

    sample_data_file_open_read = pd.read_hdf(sample_data_file_name, '/data/posterior')
    #print('Sample Data file read is {}'.format(sample_data_file_open_read))
    #print(sample_data_file_open_read.head()) # head of the DataFrame.
    #print(sample_data_file_open_read.columns) # columns of Datafrma.
    sample_data_file_read_column_mass_1 = sample_data_file_open_read.loc[:, 'mass_1': 'mass_1']
    #print(sample_data_file_read_column_mass_1)
    sample_data_file_read_column_values = sample_data_file_read_column_mass_1.values
    #print(sample_data_file_read_column_values)

    ########################################################
    # ''' Reading the SNR Value for the Injection Value'''#
    ########################################################

    snr_value_data_file_name = os.path.join(current_direc_data_file, 'StdOut_' + str(i))
    #print(snr_value_data_file_name)
    snr_value_data_file_read = open(snr_value_data_file_name, 'r').read()
    #print(snr_value_data_file_read)
    ## Omiting the all character before SNR value in the file we will use
    snr_value_data_file_value = snr_value_data_file_read[88:]
    #print(snr_value_data_file_value)

    ###################################################
    #  ''' Reading and Plotting Injection Data '''    #
    ###################################################
    injection_parameters_data_file_name_npy = os.path.join(current_direc_data_file,
                                         'injection_parameters_'+str(i)+'.npy')  ## This will call the injection parameters data file with ext '.npy'.
    #print('Injection Data File Name is  {}'.format(injection_parameters_data_file_name_npy))

    #injection_data_file_name_txt = os.path.join(current_direc_data_file, 'injection_parameters.txt') ## This will call the injection parameters data file with ext '.npy'.
    #print('Injection Data File Name is  {}'.format(injection_data_file_name_txt))

    injection_parameters_data_file_load = np.load(injection_parameters_data_file_name_npy)
    #print(injection_parameters_data_file_load)
    #print(injection_parameters_data_file_load.item())
    #print(injection_parameters_data_file_load.item().keys())
    #print(injection_parameters_data_file_load.item().values())

    injection_parameters_data_values = list(injection_parameters_data_file_load.item().values())
    #print(injection_parameters_data_values)
    injection_parameters_data_values_mass_1 = injection_parameters_data_values[12:13]  ## this will print mass_1
    #print(injection_parameters_data_values_mass_1)
    
    ##################################################
    #''' Subtracting ((M_sample  - M_injected ) ''' #
    ##################################################

    subtracted_mass_1_value = (sample_data_file_read_column_values - injection_parameters_data_values_mass_1)
    #print('Subtracted mass_1_value are {}'.format(subtracted_mass_1_value))

    font = {'family': 'serif','color': 'darkred', 'weight': 'normal', 'size' : '16'}

    sns.violinplot(x = subtracted_mass_1_value, orient='h', inner = 'box' , palette = 'RdBu_r')
    plt.title("$(M_{{sample}}  - M_{{injected}})$ for Injection Signal {} ".format(i), fontdict = font)
    text_patch = mpatches.Patch(color = 'None', label = "$M_{{injection}}$ = {} , {} "
                                .format(injection_parameters_data_values_mass_1, snr_value_data_file_value ), linewidth = 0.01)
    plt.legend(handles = [text_patch], loc = 'upper center',  prop={'size':7})   ## fontsize =12,  ncol=2, borderpad = 0.1, labelspacing =0, handleheight =0.1,
    #plt.text(-1.08,0,"$(M_{{injection}}$ = {} for signal {}".format(injection_parameters_data_values_mass_1,i), fontsize=10) ## will  not work
    #plt.legend()
    plt.grid(True)
    plt.savefig("subtracted_mass_1_value_{}.png".format(i))
    plt.show()
    plt.close()

    sns.violinplot(x='mass_1', data=sample_data_file_open_read, orient='h', palette="Set2", inner='box',hue='mass_1')
    plt.title("M_sample for injection {}".format(i))
    text_patch = mpatches.Patch(color='None', label="$M_{{injection}}$ = {} , {} "
                                .format(injection_parameters_data_values_mass_1, snr_value_data_file_value), linewidth = 0.01)
    plt.legend(handles=[text_patch],  loc = 'upper center', prop={'size':7}  )  ## fontsize = 8, ncol=2, borderpad = 0.1, labelspacing = 0, handleheight =0.1
    #plt.text(x, y, "$(M_{{injection}}$ = {} for signal {}".format(injection_parameters_data_values_mass_1, i),fontsize=10) ## will not work
    plt.grid(True)
    plt.savefig("sample_mass_1_value_{}.png".format(i))
    plt.show()
    plt.close()

    i+=1


#sample_data_file_open_read.close()
#sample_data_file_name.close()
