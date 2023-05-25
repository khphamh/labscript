'''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

def animate(i):


    plt.cla()
    a = np.loadtxt('data.txt')
    color_plot = plt.imshow(a, cmap = 'RdGy')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout()
plt.show()'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
fig = plt.figure()
ax = fig.add_subplot(111)

# I like to position my colorbars this way, but you don't have to
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')




def animate(i):
    arr = np.loadtxt('colorplot.txt')
    vmax     = np.max(arr)
    vmin     = np.min(arr)
    levels   = np.linspace(vmin, vmax, 200, endpoint = True)
    cf = ax.contourf(arr, vmax=vmax, vmin=vmin, levels=levels)
    cax.cla()
    fig.colorbar(cf, cax=cax)
    #tx.set_text('Frame {0}'.format(i))
    time.sleep(1)

ani = animation.FuncAnimation(fig, animate, interval = 1)

plt.show()