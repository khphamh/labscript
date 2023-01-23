from labscript import *

#from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.ZCU4 import ZCU4
from labscript_devices.PiezoEO import PiezoEO, PiezoEODDS
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

from labscript import *
from labscript_devices.PulseBlasterUSB import PulseBlasterUSB
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

#from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

PulseBlasterUSB(name= 'pb', loop_number = 1, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 3")
ClockLine(name = "pb_clockline_1", pseudoclock = pb.pseudoclock, connection = "flag 4")

NI_PCIe_6343(name = 'Dev1',
    parent_device = pb_clockline,
    clock_terminal = '/Dev1/PFI5',
    MAX_name = 'Dev1',
    stop_order = -1,
    acquisition_rate = 1e5
    )
    
CounterIn("counter", Dev1, connection = "ctr2", CPT_connection = "PFI13", trigger = "PFI4", numIterations = 1)

#ZCU4(name='pb')
#ClockLine(name = "pin0", pseudoclock=pb.pseudoclock, connection = "flag 1")
PiezoEO(name = 'EO', parent_device = pb_clockline_1)


#ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
#DigitalOut('pb_0',pb.direct_outputs, 'flag 0')

#DigitalOut('pb_1',pb.direct_outputs, 'flag 1')

#DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

#DigitalOut('pb_3',pb.direct_outputs, 'flag 3')
#AnalogOut('anaout_0', srs, 'ao0')
PiezoEODDS('Piezo', EO, 'a')
t = 0 
add_time_marker(t, "Start", verbose = True)
start()
counter.acquire(label = 'count', start_time = t, end_time =t+1, sample_freq = 1e5)

Piezo.setamp(t, 56)

t+=2


stop(t)
