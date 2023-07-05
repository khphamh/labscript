from labscript import *

#from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.ZCU4 import ZCU4, ZCU4DDS, ZCU4TTL

ZCU4(name='pb', com_port = 'COM5')

#ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
DigitalOut('pb_0',pb.direct_outputs, 'flag 0')

DigitalOut('pb_1',pb.direct_outputs, 'flag 1')

DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

DigitalOut('pb_3',pb.direct_outputs, 'flag 3')
ZCU4DDS('DDS6', pb.direct_outputs, '6')
ZCU4TTL('TTL', pb.direct_outputs, 't')
t = 0 
add_time_marker(t, "Start", verbose = True)
start()

#DDS6.add_pulse(6, 'const', t, 100, 30000, 100, 0, 'oneshot', 'product', '[]')
DDS6.add_pulse(6, 'const', t, 10, 3000, 100, 0, 'oneshot', 'product', '[]')
TTL.add_TTL(1, t, t + 5*(10**(-9)))
for i in range(100):
    pb_1.go_high(t)
    #pb_3.go_high(t)
    #t+=(10**(-6))/2
    t+=2*(10**(-9))
    pb_1.go_low(t)
    #pb_3.go_low(t)
    t+=(10**(-6))/2
#DDS6.add_pulse(6, 'const', t, 50, 32000, 50, 0, 'oneshot', 'product', '[]')
#DDS6.add_pulse(5, 'const', t+2*(10**(-6)), 50, 32000, 50, 0, 'oneshot', 'product', '[]')

t+=10**(-6)

stop(t)




'''

pb_0.go_high(t)

t += 100

t+=100
pb_1.go_high(t)

t+=100
pb_1.go_low(t)
'''










'''

from labscript import * #add_time_marker, AnalogOut, DigitalOut
from labscript_devices.ZCU111USB import ZCU111USB
from labscript_devices.PrawnBlaster.labscript_devices import PrawnBlaster

from labscript_devices.ZCU111.labscript_devices import ZCU111 

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
#from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

# Use a virtual, or 'dummy', device for the psuedoclock
#DummyPseudoclock(name='pseudoclock')
ZCU111(name = 'fpga')
#PrawnBlaster(name='prawn', com_port='COM6', num_pseudoclocks=1)

DigitalOut('d0', fpga.direct_outputs, connection = 'flag 2' )

t = 0 
add_time_marker(t, "Start", verbose = True)
start()

d0.go_high(t)
t += 1

t+= 1

t+=1

t+=1

t+= 1


stop(t)
'''