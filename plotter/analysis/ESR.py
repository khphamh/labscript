import numpy as np
import os
import time as time
import h5py
import pickle 

directory = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/ESR'

newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
for i in range(3):
    newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)
for file_a in os.listdir(newest_path):
    if file_a[-2:] == 'h5': #speed this up by placing all txt files in a separate location
        h5_path = newest_path + '\\' + file_a
        break
with h5py.File(h5_path,'r') as hdf5_file:
    parameters = hdf5_file['globals/ESR_parameters']
    N = np.int32(parameters.attrs['N_data_points'])
    repetitions = np.int32(parameters.attrs['repetitions'])
    freq_center = np.float64(parameters.attrs['ESR_freq_center'])
    freq_dev = np.float64(parameters.attrs['freq_dev'])
total = np.zeros(N)
used_files = []
file_list = []
xList = np.linspace(freq_center - freq_dev/(1e6), freq_center + freq_dev/(1e6), N )
np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/x_values.txt', xList)
plot_parameters_dict = {'titles': ["ESR Sequence"], 'xlabels': ["Freq (MHz)"], 'ylabels': ["Counts"], "labels": [''] }

with open('C:/Users/Yao Lab/labscript-suite/plotter/plot_parameters.txt', 'wb') as f:
    pickle.dump(plot_parameters_dict, f)

while True:
    for file_a in os.listdir(newest_path):
        if file_a[-2:] != 'h5' and file_a not in used_files: #speed this up by placing all txt files in a separate location
            file_list.append(newest_path + '\\' + file_a)
    for file_b in file_list:
        if file_b not in used_files:
            try:
                data = np.loadtxt(file_b)
                data = np.reshape(data, (2*repetitions,N))
                data[1::2, :] = data[1::2, ::-1]
                total += np.sum(data, axis = 0)
                used_files.append(file_b)
                np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/analysis_data.txt', total/(2*repetitions*len(used_files)))
                logger_file = open('C:/Users/Yao Lab/labscript-suite/plotter/logger_help.txt', 'w')
                logger_file.write(str(used_files))
                logger_file.close()
            except:
                '''logger_file = open('C:/Users/Yao Lab/labscript-suite/plotter/logger_help.txt', 'w')
                logger_file.write('ERROR')
                logger_file.close()'''
                time.sleep(1)
                pass

