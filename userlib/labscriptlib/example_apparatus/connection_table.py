# from labscript import *

# #from labscript_devices.PulseBlaster import PulseBlaster
# from labscript_devices.ZCU4 import ZCU4, ZCU4DDS

# ZCU4(name='pb')

# #ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
# DigitalOut('pb_0',pb.direct_outputs, 'flag 0')

# DigitalOut('pb_1',pb.direct_outputs, 'flag 1')

# DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

# DigitalOut('pb_3',pb.direct_outputs, 'flag 3')
# ZCU4DDS('DDS6', pb.direct_outputs, '6')

######## Emily's connection table #############
'''from labscript import *
from labscript_devices.PulseBlasterUSB import PulseBlasterUSB
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

PulseBlasterUSB(name= 'pb', loop_number = 2, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 3")

NI_PCIe_6343(name = 'Dev1',
    parent_device = pb_clockline,
    clock_terminal = '/Dev1/PFI5',
    MAX_name = 'Dev1',
    stop_order = -1,
    acquisition_rate = 1e5
    )

DigitalOut('pb_1',pb.direct_outputs, 'flag 1')



AnalogOut('anaout_0', Dev1, 'ao0')
AnalogOut('anaout_1', Dev1, 'ao1')
AnalogOut('anaout_2', Dev1, 'ao2')
AnalogOut('anaout_3', Dev1, 'ao3')


CounterIn("counter", Dev1, connection = "ctr2", CPT_connection = "PFI13", trigger = "PFI4")
#ctr 1 is cpt

DigitalOut('daq_dout_8', Dev1, 'port0/line8') 
DigitalOut('daq_dout_9', Dev1, 'port0/line9') 
'''
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
if __name__ == '__main__':
    # Begin issuing labscript primitives
    # start() elicits the commencement of the shot
    start()

    # Stop the experiment shot with stop()
    stop(1.0)

