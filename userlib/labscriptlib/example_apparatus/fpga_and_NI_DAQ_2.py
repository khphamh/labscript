from labscript import *

#from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.ZCU4 import ZCU4
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343


ZCU4(name='pb')
ClockLine(name = "pin0", pseudoclock=pb.pseudoclock, connection = "flag 7")

NI_PCIe_6343(name = 'Dev1',
    parent_device = pin0,
    clock_terminal = '/Dev1/PFI4',
    MAX_name = 'Dev1',
    stop_order = -1,
    acquisition_rate = 1e5
    )


#ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
DigitalOut('pb_0',pb.direct_outputs, 'flag 0')

DigitalOut('pb_1',pb.direct_outputs, 'flag 1')

DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

DigitalOut('pb_3',pb.direct_outputs, 'flag 3')

AnalogOut('anaout_0', Dev1, 'ao0')
AnalogOut('anaout_1', Dev1, 'ao1')
AnalogOut('anaout_2', Dev1, 'ao2')
AnalogOut('anaout_3', Dev1, 'ao3')
AnalogIn('anain_0', Dev1, 'ai1')
AnalogIn("counter", Dev1, "Dev1/ctr0")

t = 0 
add_time_marker(t, "Start", verbose = True)
start()
counter.acquire(label='measurement1', start_time=0, end_time=1000)
anaout_3.constant(t, 0)


for i in range(100):
    pb_0.go_high(t)
    pb_1.go_high(t)
    t+=5000
    pb_0.go_low(t)
    pb_1.go_low(t)
    t+=5000
    pb_0.go_high(t)
    pb_1.go_high(t)
    t+=10000
    pb_0.go_low(t)
    pb_1.go_low(t)
    t+=10000

anaout_3.constant(t, 2)
t+= 1
anaout_3.constant(t, 3)

t+=1
anaout_3.constant(t, 3)

t+=1
anaout_3.constant(t, 0)

t+= 1
anaout_3.sine_ramp(t, 1, 1,3,10)

t+= 1
anaout_3.constant(t, 2)

t+= 2

anaout_3.constant(t, 0)

t+= 10


stop(t)

















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