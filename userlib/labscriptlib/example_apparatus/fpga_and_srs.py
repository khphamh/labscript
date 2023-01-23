from labscript import *

#from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.ZCU4 import ZCU4
from labscript_devices.SRS384 import SRS384, SRS384DDS
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

from labscript import *
from labscript_devices.PulseBlasterUSB import PulseBlasterUSB
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

#from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

PulseBlasterUSB(name= 'pb', loop_number = 1, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 3")

#ZCU4(name='pb')
#ClockLine(name = "pin0", pseudoclock=pb.pseudoclock, connection = "flag 1")
SRS384(name = 'SRS', parent_device = pb_clockline)


#ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
#DigitalOut('pb_0',pb.direct_outputs, 'flag 0')

#DigitalOut('pb_1',pb.direct_outputs, 'flag 1')

#DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

#DigitalOut('pb_3',pb.direct_outputs, 'flag 3')
#AnalogOut('anaout_0', srs, 'ao0')
SRS384DDS('SRSDDS', SRS, 'a')

t = 0 
add_time_marker(t, "Start", verbose = True)
start()

SRSDDS.setamp(t, 1)
SRSDDS.setfreq(t, 400)
SRSDDS.enable_IQ(t)

t+=1


stop(t)
