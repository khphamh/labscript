from lyse import *
from pylab import *
import os
from matplotlib.widgets import TextBox, CheckButtons, Button
import h5py
import time

def submit(expression):
    print(expression)
def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    with h5py.File(globals_file, 'r+') as hdf5_file:
        group = hdf5_file['globals/raster_parameters']
        group.attrs['V_x_max'] = str(float64(group.attrs['V_x_max']) + event.xdata)
        group.attrs['V_x_min'] = str(float64(group.attrs['V_x_min']) + event.xdata)
        group.attrs['V_y_max'] = str(float64(group.attrs['V_y_max']) + event.ydata)
        group.attrs['V_y_min'] = str(float64(group.attrs['V_y_min']) + event.ydata)
        V_x_fix_box.set_val(round(event.xdata,6))
        V_y_fix_box.set_val(round(event.ydata,6))
globals_file = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/raster_scan_globals.h5'

fig, (ax1) = subplots(ncols=1)
class Index:
    def Replot(self, event):

        directory = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/raster-scan'

        newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
        for i in range(4):
            newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)
            print(newest_path)
        ser = data(newest_path)
        V_x_max = ser['V_x_max']
        V_x_min = ser['V_x_min']
        V_y_max = ser['V_y_max']
        V_y_min = ser['V_y_min']
        start = 0
        list_of_lists = []
        with h5py.File(newest_path,'r') as hdf5_file:
            group = hdf5_file['data/counter']
            for i in range(0,ser['x_points']):
                list_values = []
                for j in range(0,ser['y_points']):
                    list_values.append(group["count_" +str(i) +"_" + str(j) +"_"+str(start)][-1]-group["count_" +str(i) +"_" + str(j) +"_"+str(start)][0])
                    start+= 1
                if (i % 2):
                    list_of_lists.append(list_values)
                else:
                    list_of_lists.append(list_values[::-1])
        close('all')
        cla()
        clf()
        z_2d = array(list_of_lists)/(3*ser['dt'])
        color_plot = ax1.imshow(z_2d.T, extent=[V_x_min, V_x_max, V_y_max, V_y_min], cmap = 'RdGy')
        cb = fig.colorbar(color_plot)

cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
fig.text(0.1, 0.8 , 'From')
fig.text(0.15, 0.8 , 'To')
fig.text(0.2, 0.8 , 'Points')

fig.text(0.082, 0.755 , 'Vx')
fig.text(0.082, 0.705, 'Vy')
fig.text(0.082, 0.655, 'Vz')

fig.text(0.245, 0.755 , 'Fix Vx:')
fig.text(0.245, 0.705 , 'Fix Vy:')
fig.text(0.245, 0.655 , 'Fix Vz:')
fig.text(0.245, 0.605 , 'Fix time:')
callback = Index()

'''directory = 'C:/Users/Yao Lab/labscript-suite/Experiments/example_apparatus/raster-scan'

newest_path = min([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
for i in range(4):
    newest_path = max([os.path.join(newest_path,d) for d in os.listdir(newest_path)], key=os.path.getmtime)
    print(newest_path)
ser = data(newest_path)
V_x_max = ser['V_x_max']
V_x_min = ser['V_x_min']
V_y_max = ser['V_y_max']
V_y_min = ser['V_y_min']
start = 0
list_of_lists = []
with h5py.File(newest_path,'r') as hdf5_file:
    group = hdf5_file['data/counter']
    for i in range(0,ser['x_points']):
        list_values = []
        for j in range(0,ser['y_points']):
            list_values.append(group["count_" +str(i) +"_" + str(j) +"_"+str(start)][-1]-group["count_" +str(i) +"_" + str(j) +"_"+str(start)][0])
            start+= 1
        if (i % 2):
            list_of_lists.append(list_values)
        else:
            list_of_lists.append(list_values[::-1])
close('all')
cla()
clf()
z_2d = array(list_of_lists)/(3*ser['dt'])
fig, (ax1) = subplots(ncols=1)
color_plot = ax1.imshow(z_2d.T, extent=[V_x_min, V_x_max, V_y_max, V_y_min], cmap = 'RdGy')

fig.colorbar(color_plot)'''


V_x_min_axbox = fig.add_axes([0.092, 0.75, 0.03, 0.02])
V_x_min_box = TextBox(V_x_min_axbox, "", textalignment="center")
V_x_min_box.on_submit(submit)

V_x_max_axbox = fig.add_axes([0.142, 0.75, 0.03, 0.02])
V_x_max_box = TextBox(V_x_max_axbox, "", textalignment="center")
V_x_max_box.on_submit(submit)

x_points_axbox = fig.add_axes([0.192, 0.75, 0.03, 0.02])
x_points_box = TextBox(x_points_axbox, "", textalignment="center")
x_points_box.on_submit(submit)

V_x_fix_axbox = fig.add_axes([0.292, 0.75, 0.03, 0.02])
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

V_y_fix_axbox = fig.add_axes([0.292, 0.7, 0.03, 0.02])
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

V_z_fix_axbox = fig.add_axes([0.292, 0.65, 0.03, 0.02])
V_z_fix_box = TextBox(V_z_fix_axbox, "", textalignment="center")
V_z_fix_box.on_submit(submit)
V_z_fix_box.set_val(50)

time_fix_axbox = fig.add_axes([0.292, 0.6, 0.03, 0.02])
time_fix_box = TextBox(time_fix_axbox, "", textalignment="center")
time_fix_box.on_submit(submit)
time_fix_box.set_val(0.001)

CPS_axbox = fig.add_axes([0.092, 0.5, 0.15, 0.04])
CPS_box = TextBox(CPS_axbox, "", textalignment="center")
CPS_box.on_submit(submit)

ReplotButton = Button(fig.add_axes([0.05, 0.8, 0.045, 0.02]), 'Plot Newest')
ReplotButton.on_clicked(callback.Replot)
FixButton = Button(fig.add_axes([0.29, 0.8, 0.03, 0.02]), 'Fix')
RunCPSButton = Button(fig.add_axes([0.09, 0.45, 0.03, 0.02]), 'Run CPS')
StopCPSButton = Button(fig.add_axes([0.2, 0.45, 0.03, 0.02]), 'Stop CPS')