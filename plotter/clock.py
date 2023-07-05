
import matplotlib
import os
import subprocess
import psutil
import numpy as np
import time
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import style

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageGrab


LARGE_FONT= ("Verdana", 12)
style.use("ggplot")

f = Figure(figsize=(5,5), dpi=100)
f_2 = Figure(figsize=(5,5), dpi=100)

a = f.add_subplot(111)
b = f_2.add_subplot(111)
b.axis('equal')
div = make_axes_locatable(b)
cax = div.append_axes('right', '5%', '5%')
b.set_facecolor('white')
a.set_facecolor('white')
for word in ['left', 'right', 'top', 'bottom']:
    b.spines[word].set_color('black')
    a.spines[word].set_color('black')
b.tick_params(axis='x', colors='black')    #setting up X-axis tick color to red
b.tick_params(axis='y', colors='black')    #setting up X-axis tick color to red
a.tick_params(axis='x', colors='black')    #setting up X-axis tick color to red
a.tick_params(axis='y', colors='black')    #setting up X-axis tick color to red


def animate(i):
    yList = np.loadtxt('analysis_data.txt')
    a.clear()
    a.plot(yList)

class Data_Analysis(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Analysis")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)



        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):

            frame = F(container, self)
            

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="Graph Page",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):        
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Scan Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        self.canvas = FigureCanvasTkAgg(f_2, self)
        #self.canvas.draw()
        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH)

        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        #self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().config(width = 600, height = 600)

        self.canvas._tkcanvas.place(x = 900, y = 100)
        
        #self.canvas.get_tk_widget().configure(bg='white')
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Plot Newest",
                            command=self.plot_newest)
        button2.pack()

        button2 = ttk.Button(self, text="Plot Loaded",
                            command=self.plot_loaded)
        button2.pack()

        button3 = ttk.Button(self, text="Load",
                            command=self.openfile)
        button3.pack()
        
        x_offset = 0
        y_offset = 0 
        fix_button = ttk.Button(self, text="Fix",
                            command=self.fix)
        fix_button.place(x = x_offset + 460, y = y_offset + 150)

    
        scan_button = ttk.Button(self, text="Scan",command=self.scan)
        scan_button.place(x = x_offset + 460, y = y_offset + 400)

        stop_button = ttk.Button(self, text="Stop",command=self.stop)
        stop_button.place(x = x_offset + 370, y = y_offset + 400)

        run_CPS_button = ttk.Button(self, text="Run CPS",command=lambda: self.run_CPS(True))
        run_CPS_button.place(x = x_offset + 160, y = y_offset + 500)

        stop_CPS_button = ttk.Button(self, text="Stop CPS",command=lambda: self.run_CPS(False))
        stop_CPS_button.place(x = x_offset + 260, y = y_offset + 500)

        self.CPS_label = ttk.Label(self, background='purple', foreground='white')
        self.CPS_label.place(x = x_offset + 260, y = y_offset + 450)

        self.V_x_min_text = tk.Entry(self, width = 7)
        self.V_x_min_text.place(x = x_offset + 160, y = y_offset + 200)
        self.V_x_min_text.insert(tk.END, "-0.1")

        self.V_x_max_text = tk.Entry(self, width = 7)
        self.V_x_max_text.place(x = x_offset + 260, y = y_offset + 200)
        self.V_x_max_text.insert(tk.END, "0.1")

        self.V_x_points_text = tk.Entry(self, width = 7)
        self.V_x_points_text.place(x = x_offset + 360, y = y_offset + 200)
        self.V_x_points_text.insert(tk.END, "100")

        self.Fix_V_x_text = tk.Checkbutton(self, text = "Fix Vx")
        self.Fix_V_x_text.place(x = x_offset + 400, y = 200)

        self.V_x_text = tk.Entry(self, width = 7)
        self.V_x_text.place(x = x_offset + 460, y = y_offset + 200)

        self.V_y_min_text = tk.Entry(self, width = 7)
        self.V_y_min_text.place(x = x_offset + 160, y = y_offset + 250)
        self.V_y_min_text.insert(tk.END, "-0.1")

        self.V_y_max_text = tk.Entry(self, width = 7)
        self.V_y_max_text.place(x = x_offset + 260, y = y_offset + 250)
        self.V_y_max_text.insert(tk.END, "0.1")

        self.V_y_points_text = tk.Entry(self, width = 7)
        self.V_y_points_text.place(x = x_offset + 360, y = y_offset + 250)
        self.V_y_points_text.insert(tk.END, "100")

        self.Fix_V_y_text = tk.Checkbutton(self, text = "Fix Vy")
        self.Fix_V_y_text.place(x = x_offset + 400, y = 250)

        self.V_y_text = tk.Entry(self, width = 7)
        self.V_y_text.place(x = x_offset + 460, y = y_offset + 250)

        self.V_z_min_text = tk.Entry(self, width = 7)
        self.V_z_min_text.place(x = x_offset + 160, y = y_offset + 300)
        self.V_z_min_text.insert(tk.END, "0")

        self.V_z_max_text = tk.Entry(self, width = 7)
        self.V_z_max_text.place(x = x_offset + 260, y = y_offset + 300)
        self.V_z_max_text.insert(tk.END, "100")

        self.V_z_points_text = tk.Entry(self, width = 7)
        self.V_z_points_text.place(x = x_offset + 360, y = y_offset + 300)
        self.V_z_points_text.insert(tk.END, "100")

        self.Fix_V_z_text = tk.Checkbutton(self, text = "Fix Vz")
        self.Fix_V_z_text.place(x = x_offset + 400, y = 300)

        self.V_z_text = tk.Entry(self, width = 7)
        self.V_z_text.place(x = x_offset + 460, y = y_offset + 300)
        self.V_z_text.insert(tk.END, "50")

        self.dt_min_text = tk.Entry(self, width = 7)
        self.dt_min_text.place(x = x_offset + 160, y = y_offset + 350)
        self.dt_min_text.insert(tk.END, "0.001")

        self.dt_max_text = tk.Entry(self, width = 7)
        self.dt_max_text.place(x = x_offset + 260, y = y_offset + 350)
        self.dt_max_text.insert(tk.END, "0.003")

        self.dt_points_text = tk.Entry(self, width = 7)
        self.dt_points_text.place(x = x_offset + 360, y = y_offset + 350)
        self.dt_points_text.insert(tk.END, "10")

        self.Fix_dt_text = tk.Checkbutton(self, text = "Fix dt")
        self.Fix_dt_text.place(x = x_offset + 400, y = 350)

        self.dt_text = tk.Entry(self, width = 7)
        self.dt_text.place(x = x_offset + 460, y = y_offset + 350)
        self.dt_text.insert(tk.END, "0.001")

        self.x_cursor = None
        self.y_cursor = None
        self.cursor = None

        f_2.canvas.mpl_connect('button_press_event', self.mouse_event)

    def mouse_event(self,event):
        print('x: {} and y: {}'.format(event.xdata, event.ydata))
        self.x_cursor = event.xdata
        self.y_cursor = event.ydata
        if self.cursor is not None:
            self.cursor.remove()
        if self.x_cursor is not None:
            self.cursor = b.scatter(self.x_cursor, self.y_cursor, s = 5, c = 'red', marker = '+')
        self.canvas.draw()
        self.V_x_text.delete(0, tk.END)
        self.V_y_text.delete(0, tk.END)

        self.V_x_text.insert(tk.END, str(self.x_cursor))
        self.V_y_text.insert(tk.END, str(self.x_cursor))

    def plot_newest(self):
        print("Plotting Newest Image")
        arr = np.loadtxt('colorplot.txt')
        vmax     = np.max(arr)
        vmin     = np.min(arr)
        levels   = np.linspace(vmin, vmax, 200, endpoint = True)
        cf = b.contourf(arr, levels=levels)
        cax.cla()
        b.set_xlim([0, len(arr[0])-1])
        b.set_ylim([0, len(arr[0])-1])
        f_2.colorbar(cf, cax=cax)
        self.canvas.draw()

    def plot_loaded(self):
        print("Plotting Loaded Image")
        arr = np.loadtxt(self.loaded_file)
        vmax     = np.max(arr)
        vmin     = np.min(arr)
        levels   = np.linspace(vmin, vmax, 200, endpoint = True)
        cf = b.contourf(arr, vmax=vmax, vmin=vmin, levels=levels)
        cax.cla()
        b.set_xlim([0, len(arr[0])-1])
        b.set_ylim([0, len(arr[0])-1])
        f_2.colorbar(cf, cax=cax)
        self.canvas.draw()
        
    def openfile(self):
        self.loaded_file = tk.filedialog.askopenfilename()
        return 0

    def get_values(self):
        V_x_min, V_x_max, V_x_points, V_x = self.V_x_min_text.get(), self.V_x_max_text.get(), self.V_x_points_text.get(), self.V_x_text.get()
        V_y_min, V_y_max, V_y_points, V_y = self.V_y_min_text.get(), self.V_y_max_text.get(), self.V_y_points_text.get(), self.V_y_text.get()
        V_z_min, V_z_max, V_z_points, V_z = self.V_z_min_text.get(), self.V_z_max_text.get(), self.V_z_points_text.get(), self.V_z_text.get()
        dt_min, dt_max, dt_points, dt = self.dt_min_text.get(), self.dt_max_text.get(), self.dt_points_text.get(), self.dt_text.get()
        return (V_x_min, V_x_max, V_x_points, V_x), (V_y_min, V_y_max, V_y_points, V_y), (V_z_min, V_z_max, V_z_points, V_z), (dt_min, dt_max, dt_points, dt)

    def fix(self):
        print(self.get_values())
        return self.get_values()

    def scan(self):
        return 0

    def stop(self):
        return 0

    def run_CPS(self, enable):
        self.cancel = not enable
        self.CPS_updater()
        

    def CPS_updater(self):
        if not self.cancel:
            num_string = np.loadtxt('rand_num.txt')
            self.CPS_label.config(text=num_string)
            self.CPS_ID = self.CPS_label.after(1000, self.CPS_updater)
        else:
            self.CPS_ID = self.CPS_label.after_cancel(self.CPS_updater)

        #else:
            #self.CPS_label.after_cancel(self.CPS_ID)


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Data Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.canvas = FigureCanvasTkAgg(f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().configure(bg='white')

        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button2 = ttk.Button(self, text="Load Analysis Script",
                            command=self.loadfile)
        button2.pack()

        button3 = ttk.Button(self, text="Run Analysis Script",
                            command=self.runfile)
        button3.pack()


    def loadfile(self):
        self.loaded_file = tk.filedialog.askopenfilename()
        '''for proc in psutil.process_iter():
            try:
                # Get process name & pid from process object.
                processName = proc.name()
                processID = proc.pid

                if processName.startswith("python3"): # adapt this line to your needs
                    print(f"I will kill {processName}[{processID}] : {''.join(proc.cmdline())})")
                    # proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                print(e)'''
        try:
            self.subprocess.kill()
        except:
            print("no subprocess")
            pass
    def runfile(self):
        print(self.loaded_file)
        '''with open(self.loaded_file) as f:
            exec(f.read())'''
        my_pid = os.getpid()
        self.subprocess = subprocess.Popen(['python', self.loaded_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


app = Data_Analysis()
#app.geometry("1280x720")

ani = animation.FuncAnimation(f, animate, interval=5000)

#ani = animation.FuncAnimation(f_2, animate, interval=1000)

app.mainloop()
