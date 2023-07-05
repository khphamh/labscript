from lyse import *
from pylab import *


import h5py
df = data()
print(df['filepath'][0])
'''start = 0
list_of_lists = []
with h5py.File(path,'r') as hdf5_file:
    group = hdf5_file['data/counter']
    for i in range(0,50):
        list_values = []
        for j in range(0,50):
            list_values.append(group["count_" +str(i) +"_" + str(j) +"_"+str(start)][-1]-group["count_" +str(i) +"_" + str(j) +"_"+str(start)][0])
            start+= 1
        if (i % 2):
            list_of_lists.append(list_values)
        else:
            list_of_lists.append(list_values[::-1])
            
z_2d = array(list_of_lists)

imshow(z_2d.T,  cmap = 'RdGy')

colorbar()'''