import numpy as np
import time 

directory = 'C:/Users/nateu/Downloads/plotter'

total = np.random.rand(1)
while True:
    total = np.random.rand(1)
    np.savetxt('C:/Users/nateu/Downloads/plotter/rand_num.txt', total)
    time.sleep(0.5)