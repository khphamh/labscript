from labscript import *
from labscript_devices.PulseBlasterUSB import PulseBlasterUSB
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

#from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
#from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

# Use a virtual, or 'dummy', device for the psuedoclock
#DummyPseudoclock(name='pseudoclock', BLACS_connection = 'dummy')
PulseBlasterUSB(name= 'pulseblaster', board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pulseblaster.pseudoclock, connection = "flag 0")

NI_PCIe_6343(name = 'Dev1',
    parent_device = pb_clockline,
    clock_terminal = '/Dev1/PFI5',
    MAX_name = 'Dev1',
    stop_order = -1,
    acquisition_rate = 1e4
    )

AnalogOut('anaout_0', Dev1, 'ao0')
AnalogOut('anaout_1', Dev1, 'ao1')
AnalogOut('anaout_2', Dev1, 'ao2')
AnalogOut('anaout_3', Dev1, 'ao3')

#WaitMonitor(name='wait_monitor', parent_device = Dev1, connection = 'port0/line0', acquisition_device = Dev1, acquisition_connection = 'ctr0', timeout_device = Dev1, timeout_connection = "PFI1")
#DigitalOut("daq_do_1", Dev1, "port0/line1")

DigitalOut('daq_dout_0', Dev1, 'port0/line8') 
DigitalOut('daq_dout_1', Dev1, 'port0/line9') 


t = 0 
#counter.acquire(label='measurement1', start_time=0, end_time=1)
#counter.acquire(label='measurement2', start_time=2, end_time=3)

add_time_marker(t, "Start", verbose = True)
start()


daq_dout_0.go_high(t)

t+= 2

daq_dout_0.go_low(t)

t+= 5

daq_dout_0.go_high(t)

t+= 2

daq_dout_0.go_low(t)

t+= 3

daq_dout_0.go_high(t)

t+= 2

daq_dout_0.go_low(t)

t+=1

"""

anaout_3.constant(t, 0)
t += 1*(10)
add_time_marker(t, "Start", verbose = True)

anaout_3.constant(t, 2)

t += 1
add_time_marker(t, "Start", verbose = True)

anaout_3.constant(t, 3)

t += 1
add_time_marker(t, "Start", verbose = True)

anaout_3.constant(t, 3)

t += 5
add_time_marker(t, "Start", verbose = True)

anaout_3.constant(t, 0)
t+=1
t = t*(10**5)
anaout_3.ramp(t, 5, 0,3,1)


t += 6
add_time_marker(t, "Start", verbose = True)

anaout_3.constant(t, 2)

t += 2
add_time_marker(t, "Start", verbose = True)

anaout_3.constant(t, 0)

t+= 10
add_time_marker(t, "Start", verbose = True)
"""
stop(t)