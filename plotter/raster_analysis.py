import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import TextBox, CheckButtons, Button
import h5py
import time
from itertools import count
from matplotlib.animation import FuncAnimation
import IPython.display as display

def submit(expression):
    print(expression)
def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    with h5py.File(globals_file, 'r+') as hdf5_file:
        group = hdf5_file['globals/raster_parameters']
        group.attrs['V_x_max'] = str(np.float64(group.attrs['V_x_max']) + event.xdata)
        group.attrs['V_x_min'] = str(np.float64(group.attrs['V_x_min']) + event.xdata)
        group.attrs['V_y_max'] = str(np.float64(group.attrs['V_y_max']) + event.ydata)
        group.attrs['V_y_min'] = str(np.float64(group.attrs['V_y_min']) + event.ydata)
        V_x_fix_box.set_val(round(event.xdata,6))
        V_y_fix_box.set_val(round(event.ydata,6))
fig, ax = plt.subplots()

class Index:
    def Replot(self, event):
        directory = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/raster-scan'

        newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
        for i in range(4):
            newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)
            #print(newest_path)

        start = 0
        list_of_lists = []
        with h5py.File(newest_path,'r') as hdf5_file:
            group = hdf5_file['data/counter']
            parameters = hdf5_file['globals/raster_parameters']
            for i in range(0,int(parameters.attrs['x_points'])):
                list_values = []
                for j in range(0,int(parameters.attrs['y_points'])):
                    list_values.append(group["count_" +str(i) +"_" + str(j) +"_"+str(start)][-1]-group["count_" +str(i) +"_" + str(j) +"_"+str(start)][0])
                    start+= 1
                if (i % 2):
                    list_of_lists.append(list_values)
                else:
                    list_of_lists.append(list_values[::-1])
            V_x_max = np.float64(parameters.attrs['V_x_max'])
            V_x_min = np.float64(parameters.attrs['V_x_min'])
            V_y_max = np.float64(parameters.attrs['V_y_max'])
            V_y_min = np.float64(parameters.attrs['V_y_min'])
            z_2d = np.array(list_of_lists)/(3*np.float64(parameters.attrs['dt']))
        color_plot = ax.imshow(z_2d.T, extent=[V_x_min, V_x_max, V_y_max, V_y_min], cmap = 'RdGy')
        cb = fig.colorbar(color_plot, ax = ax)
globals_file = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/raster_scan_globals.h5'
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
'''height_axbox = fig.add_axes([0.1, 0.9, 0.05, 0.04])
height_text_box = TextBox(height_axbox, "height", textalignment="center")
height_text_box.on_submit(submit)
height_text_box.set_val(1)'''
callback = Index()
fig.text(0.1, 0.8 , 'From')
fig.text(0.15, 0.8 , 'To')
fig.text(0.2, 0.8 , 'Points')
fig.text(0.25, 0.8 , 'Fix')

fig.text(0.082, 0.755 , 'Vx')
fig.text(0.082, 0.705, 'Vy')
fig.text(0.082, 0.6555, 'Vz')

V_x_min_axbox = fig.add_axes([0.092, 0.75, 0.03, 0.02])
V_x_min_box = TextBox(V_x_min_axbox, "", textalignment="center")
V_x_min_box.on_submit(submit)

V_x_max_axbox = fig.add_axes([0.142, 0.75, 0.03, 0.02])
V_x_max_box = TextBox(V_x_max_axbox, "", textalignment="center")
V_x_max_box.on_submit(submit)

x_points_axbox = fig.add_axes([0.192, 0.75, 0.03, 0.02])
x_points_box = TextBox(x_points_axbox, "", textalignment="center")
x_points_box.on_submit(submit)

V_x_fix_axbox = fig.add_axes([0.242, 0.75, 0.03, 0.02])
V_x_fix_box = TextBox(V_x_fix_axbox, "", textalignment="center")
V_x_fix_box.on_submit(submit)

V_y_min_axbox = fig.add_axes([0.092, 0.7, 0.03, 0.02])
V_y_min_box = TextBox(V_y_min_axbox, "", textalignment="center")
V_y_min_box.on_submit(submit)

V_y_max_axbox = fig.add_axes([0.142, 0.7, 0.03, 0.02])
V_y_max_box = TextBox(V_y_max_axbox, "", textalignment="center")
V_y_max_box.on_submit(submit)

y_points_axbox = fig.add_axes([0.192, 0.7, 0.03, 0.02])
y_points_box = TextBox(y_points_axbox, "", textalignment="center")
y_points_box.on_submit(submit)

V_y_fix_axbox = fig.add_axes([0.242, 0.7, 0.03, 0.02])
V_y_fix_box = TextBox(V_y_fix_axbox, "", textalignment="center")
V_y_fix_box.on_submit(submit)

V_z_min_axbox = fig.add_axes([0.092, 0.65, 0.03, 0.02])
V_z_min_box = TextBox(V_z_min_axbox, "", textalignment="center")
V_z_min_box.on_submit(submit)

V_z_max_axbox = fig.add_axes([0.142, 0.65, 0.03, 0.02])
V_z_max_box = TextBox(V_z_max_axbox, "", textalignment="center")
V_z_max_box.on_submit(submit)

z_points_axbox = fig.add_axes([0.192, 0.65, 0.03, 0.02])
z_points_box = TextBox(z_points_axbox, "", textalignment="center")
z_points_box.on_submit(submit)

V_z_fix_axbox = fig.add_axes([0.242, 0.65, 0.03, 0.02])
V_z_fix_box = TextBox(V_z_fix_axbox, "", textalignment="center")
V_z_fix_box.on_submit(submit)
CPS_axbox = fig.add_axes([0.092, 0.5, 0.15, 0.04])
CPS_box = TextBox(CPS_axbox, "", textalignment="center")
CPS_box.on_submit(submit)

ReplotButton = Button(fig.add_axes([0.05, 0.8, 0.045, 0.02]), 'Plot Newest')
ReplotButton.on_clicked(callback.Replot)
FixButton = Button(fig.add_axes([0.29, 0.8, 0.03, 0.02]), 'Fix')
RunCPSButton = Button(fig.add_axes([0.09, 0.45, 0.03, 0.02]), 'Run CPS')
StopCPSButton = Button(fig.add_axes([0.2, 0.45, 0.03, 0.02]), 'Stop CPS')
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

#ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.show()
