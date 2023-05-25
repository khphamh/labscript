import os
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from itertools import count
index = count()

directory = 'C:/Users/Yao-lab-3/labscript-suite/Experiments/example_apparatus/ESR'

newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
for i in range(3):
    newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)
    #print(newest_path)
list_values = np.zeros(100)
used_files = []
def animate(i):
    #print(i)
    global list_values
    plt.cla()
    #list_values = np.zeros(201)
    file_list = []
    global newest_path
    for file_a in os.listdir(newest_path):
        file_list.append(newest_path + '\\' + file_a)
    #print(newest_path)
    for file_b in file_list:
        start = 0
        if file_b not in used_files:
            print(file_b)
            try:
                with h5py.File(file_b,'r') as hdf5_file:
                    group = group = hdf5_file['data']
                    for i in range(100):
                        list_values += group['counter_data'][i*200:100+i*200]
                        list_values[::-1] += group['counter_data'][100+i*200:200+i*200]
                used_files.append(file_b)
            except:
                time.sleep(10)
                pass
    try:
        plt.xlabel("Freq (Ghz)")
        plt.ylabel("counts")
        plt.plot(np.linspace(3.3,3.6,100), list_values/(200*len(used_files)))
    except:
        pass

ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()