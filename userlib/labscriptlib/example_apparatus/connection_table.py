from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.ZCU111 import ZCU111, ZCU111DDS
#from labscript_devices.PiezoEO import PiezoEO, PiezoEODDS
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
from labscript_devices.SRS384 import SRS384, SRS384DDS

PulseBlasterESRPro500(name= 'pb', loop_number = numIterations,  extra_flags = 0, extra_inst = 3, extra_inst_data = 3, extra_length= 10, inst_location = 13, additional_inst = 8, loop_start = 1, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")
ClockLine(name = "pb_clockline_2", pseudoclock = pb.pseudoclock, connection = "flag 5")

NI_PCIe_6343(name = 'Dev2',
    parent_device = pb_clockline,
    clock_terminal = '/Dev2/PFI5',
    MAX_name = 'Dev2',
    static_AO = True,
    stop_order = -1,
    acquisition_rate = 1e5
    )
DigitalOut('pb_1',pb.direct_outputs, 'flag 1') #counter readout
DigitalOut('pb_2',pb.direct_outputs, 'flag 2') #laser 
DigitalOut('pb_3',pb.direct_outputs, 'flag 3') #MW trigger
DigitalOut('pb_4',pb.direct_outputs, 'flag 4') #MW gate

StaticAnalogOut('anaout_0', Dev2, 'ao0')
StaticAnalogOut('anaout_1', Dev2, 'ao1')

CounterIn("counter", Dev2, connection = "ctr2", CPT_connection = "PFI0", trigger = "PFI1", numIterations = numIterations)
#ctr 1 is cpt

DigitalOut('daq_dout_8', Dev2, 'port0/line8') 
DigitalOut('daq_dout_9', Dev2, 'port0/line9') 
ZCU111(name = 'ZCU', parent_device = pb_clockline_2, com_port = 'COM5')
ZCU111DDS('ZCUDDS', ZCU, 'a')
#PiezoEO(name = 'EO', parent_device = pb_clockline_2)
#PiezoEODDS('Piezo', EO, 'a')
SRS384(name = 'SRS', parent_device = pb_clockline_2, com_port = 'COM12')
SRS384DDS('SRSDDS', SRS, 'a')
if __name__ == '__main__':
    # Begin issuing labscript primitives
    # start() elicits the commencement of the shot
    start()

    # Stop the experiment shot with stop()
    stop(1.0)

