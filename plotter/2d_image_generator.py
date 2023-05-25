import numpy as np
import time

x_value = 0
i = 0
while True:
    i += 1
    #a = i*np.identity(20) + np.random.randn(20,20)
    a = np.random.rand(1,300)
    b = i*np.identity(50) + np.random.rand(50,50)
    np.savetxt('data' + str(i)+ '.txt', a)
    np.savetxt('colorplot.txt', b)
    time.sleep(3)


