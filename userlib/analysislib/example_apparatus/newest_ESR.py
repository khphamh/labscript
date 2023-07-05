from lyse import *
from pylab import *
import os
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from itertools import count
index = count()

directory = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/ESR'

newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
for i in range(3):
    newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)
    #print(newest_path)
def animate(i):
    #print(i)
    plt.cla()
    list_values = zeros(201)
    file_list = []
    for file in os.listdir(newest_path):
        file_list.append(newest_path + '\\' + file)
    for file_a in file_list:
        start = 0
        try:
            with h5py.File(file_a,'r') as hdf5_file:
                group = group = hdf5_file['data/counter']
                for i in range(201):
                    list_values[i] += (group["count_up" + str(i)+'_'+str(start)][-1] - group["count_up" + str(i)+'_'+str(start)][0])
                    start += 1
                for i in range(201, 0,-1):
                    list_values[201-i] += (group["count_down" + str(i)+'_'+str(start)][-1] - group["count_down" + str(i)+'_'+str(start)][0])
                    start += 1
        except:
            pass
    try:
        list_values = list_values/(20*len(file_list) )
        plt.plot(linspace(3,4,201), list_values)
    except:
        pass

ani = FuncAnimation(gcf(), animate,  interval=1000,cache_frame_data=False)

tight_layout()
