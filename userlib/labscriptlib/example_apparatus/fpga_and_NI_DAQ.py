from labscript import *

#from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.ZCU4 import ZCU4
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343


ZCU4(name='pb')
ClockLine(name = "pin0", pseudoclock=pb.pseudoclock, connection = "flag 1")

NI_PCIe_6343(name = 'Dev1',
    parent_device = pin0,
    clock_terminal = '/Dev1/PFI4',
    MAX_name = 'Dev1',
    stop_order = -1,
    acquisition_rate = 1e5
    )


#ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
#DigitalOut('pb_0',pb.direct_outputs, 'flag 0')

#DigitalOut('pb_1',pb.direct_outputs, 'flag 1')

DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

DigitalOut('pb_3',pb.direct_outputs, 'flag 3')

AnalogOut('anaout_0', Dev1, 'ao0')
AnalogOut('anaout_1', Dev1, 'ao1')
AnalogOut('anaout_2', Dev1, 'ao2')
AnalogOut('anaout_3', Dev1, 'ao3')
#CounterIn('anain_0', Dev1, 'ai1')

t = 0 
add_time_marker(t, "Start", verbose = True)
start()
#anain_0.acquire(label = "measurement 1", start_time = 0, end_time = 10)

"""
for i in range(100):
    pb_0.go_high(t)
    pb_1.go_high(t)
    t+=5000*10
    pb_0.go_low(t)
    pb_1.go_low(t)
    t+=5000*10
    pb_0.go_high(t)
    pb_1.go_high(t)
    t+=10000*10
    pb_0.go_low(t)
    pb_1.go_low(t)
    t+=10000*10"""
"""for i in range(10):
    pb_1.go_high(t)
    #pb_1.go_high(t)
    t+=(10**(-5))

    #pb_1.go_low(t)
    t+=(10**(-5))

    #pb_1.go_high(t)
    pb_1.go_low(t)
    t+=(10**(-5))

    #pb_1.go_low(t)
    t+=(10**(-5))"""
anaout_3.constant(t, 2)
t+= 10**(-5)
anaout_3.constant(t, 3)

t+= 2*10**(-5)

anaout_3.constant(t, 0)

t+= 10**(-5)
#anaout_3.sine_ramp(t, 10**(-5), 1,3,10)
"""
t+= 10**(-5)
anaout_3.constant(t, 2)

t+= 2*10**(-5)

anaout_3.constant(t, 0)

t+= 10*(10**(-5))"""


stop(t)
