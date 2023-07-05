import numpy as np
import os
import time as time
import h5py
import pickle 

directory = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/Rabi'
index_num = 1
newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
for i in range(3):
    newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)
for file_a in os.listdir(newest_path):
    if file_a[-2:] == 'h5': #speed this up by placing all txt files in a separate location
        h5_path = newest_path + '\\' + file_a
        break
with h5py.File(h5_path,'r') as hdf5_file:
    parameters = hdf5_file['globals/rabi_parameters']
    xList = eval(parameters.attrs['rabi_pulse_time'])[0:-1*index_num]
xList_len = len(xList) + 1
used_files = set()
file_list = set()
ref = np.zeros(xList_len)
sig = np.zeros(xList_len)
used_files_array = np.zeros(xList_len)
np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/x_values.txt',xList)
plot_parameters_dict = {'titles': ["Rabi Sequence at 125 ns counter gate", "Contrast"], 'xlabels': ["MW pulse time (ns)", "MW pulse time (ns)"], 'ylabels': ["Counts", "sig/ref"], "labels": ['[ref,sig]', " "] }

with open('C:/Users/Yao Lab/labscript-suite/plotter/plot_parameters.txt', 'wb') as f:
    pickle.dump(plot_parameters_dict, f)

while True:
    for file_a in os.listdir(newest_path):
        if file_a[-2:] != 'h5' and file_a not in used_files: #speed this up by placing all txt files in a separate location
            file_list.add(newest_path + '\\' + file_a)
    for k in range(index_num,xList_len):
        for file_b in file_list:
            if float(file_b[118:120]) == k and file_b not in used_files:
                try:
                    data = np.loadtxt(file_b)
                    ref[k-1] += data[0]
                    sig[k-1] += data[1]
                    used_files_array[k-1] += 1
                    used_files.add(file_b)
                    logger_file = open('C:/Users/Yao Lab/labscript-suite/plotter/logger_help.txt', 'w')
                    logger_file.write(str(used_files))
                    logger_file.close()
                    #total = sig[1:]/ref[1:]
                    #np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/analysis_data.txt', total)
                    np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/analysis_data.txt', np.vstack((ref[0:-1]/used_files_array[0:-1],sig[0:-1]/used_files_array[0:-1])))
                    np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/analysis_data_2.txt', (sig[0:-1]/ref[0:-1]))

                except:
                    '''logger_file = open('C:/Users/Yao Lab/labscript-suite/plotter/logger_help.txt', 'w')
                    logger_file.write('ERROR')
                    logger_file.close()'''
                    time.sleep(1)
                    pass
    file_list = set(file_list) - set(used_files)