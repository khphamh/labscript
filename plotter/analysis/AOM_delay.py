import numpy as np
import os
import time as time
directory = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/AOM_delay'

newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
for i in range(3):
    newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)


total = np.zeros(26)
used_files = []
file_list = []

while True:
    for file_a in os.listdir(newest_path):
        if file_a[-2:] != 'h5': #speed this up by placing all txt files in a separate location
            file_list.append(newest_path + '\\' + file_a)

    for file_b in file_list:
        start = 0
        if file_b not in used_files:
            try:
                data = np.loadtxt(file_b)

                total[int(file_b[128:130])] += data[-1]
                used_files.append(file_b)
                logger_file = open('C:/Users/Yao Lab/labscript-suite/plotter/logger_help.txt', 'w')
                logger_file.write(str(used_files))
                logger_file.close()
            except:
                logger_file = open('C:/Users/Yao Lab/labscript-suite/plotter/logger_help.txt', 'w')
                logger_file.write('ERROR')
                logger_file.close()
                time.sleep(10)
                pass


        np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/analysis_data.txt', total)
    np.savetxt('C:/Users/Yao Lab/labscript-suite/plotter/analysis_data.txt', total)

