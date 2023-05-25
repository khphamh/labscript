import numpy as np
import time 

directory = 'C:/Users/nateu/Downloads/plotter'

total = np.random.rand(1,300)
while True:
    total += np.random.rand(1,300)
    np.savetxt('C:/Users/nateu/Downloads/plotter/analysis_data.txt', total)
    time.sleep(3)