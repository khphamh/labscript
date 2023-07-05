from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
PulseBlasterESRPro500(name= 'pb', loop_number = numIterations,  extra_flags = 0, extra_inst = 3, extra_inst_data = 2, extra_length= 10, inst_location = 12, additional_inst = 8, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")

NI_PCIe_6343(name = 'Dev2',
    parent_device = pb_clockline,
    clock_terminal = '/Dev2/PFI5',
    MAX_name = 'Dev2',
    stop_order = -1,
    acquisition_rate = 1e5
    )
DigitalOut('pb_1',pb.direct_outputs, 'flag 1') #trigger DAQ counter
DigitalOut('pb_2',pb.direct_outputs, 'flag 2') #laser
DigitalOut('pb_3',pb.direct_outputs, 'flag 3') #trigger fpga
DigitalOut('pb_5',pb.direct_outputs, 'flag 5') #dummy

AnalogOut('anaout_0', Dev2, 'ao0')
AnalogOut('anaout_1', Dev2, 'ao1')
AnalogOut('anaout_2', Dev2, 'ao2')
AnalogOut('anaout_3', Dev2, 'ao3')
CounterIn("counter", Dev2, connection = "ctr2", CPT_connection = "PFI0", trigger = "PFI1", numIterations = numIterations)

DigitalOut('daq_dout_8', Dev2, 'port0/line8') 
DigitalOut('daq_dout_9', Dev2, 'port0/line9') 


t = 0  
add_time_marker(t, "Start", verbose = True)
start()
#dt = 5e-3
counter_gate_time = 1e-6
counter.fast_counter(1e8,2)
#for i in range(10):
#t += 20e-9
'''pb_2.go_high(t)
t += AOM_delay
pb_1.go_high(t)
t += counter_gate_time
pb_1.go_low(t)
t+= 1e-6

pb_2.go_low(t)
pb_3.go_high(t)
t+= 20e-9
pb_3.go_low(t)
t+=20e-6

pb_2.go_high(t)
t += AOM_delay
pb_1.go_high(t)
t += counter_gate_time
pb_1.go_low(t)
t+= 1e-6

pb_2.go_low(t)
t+= 1e-6'''

if AOM_delay + counter_gate_time == 3e-6:
    pb_1.go_high(AOM_delay - 12e-9)
    pb_1.go_low(AOM_delay+counter_gate_time - 12e-9)
else:
    pb_1.go_high(AOM_delay)
    pb_1.go_low(AOM_delay+counter_gate_time)
pb_2.go_high(3e-6)
pb_2.go_low(9e-6)
t+= 10e-6

if AOM_delay + counter_gate_time == 3e-6:
    pb_1.go_high(t + AOM_delay - 12e-9)
    pb_1.go_low(t+AOM_delay+counter_gate_time - 12e-9)
else:
    pb_1.go_high(t+AOM_delay)
    pb_1.go_low(t+AOM_delay+counter_gate_time)
pb_2.go_high(t+3e-6)
pb_2.go_low(t+9e-6)
t+= 10e-6


pb_1.go_high(t)
t += 6e-8
pb_1.go_low(t)
t+= 6e-8
pb_1.go_high(t)
t += 6e-8
pb_1.go_low(t)
t+= 6e-8
pb_1.go_high(t)
t += 6e-8
pb_1.go_low(t)
t+= 6e-8
pb_1.go_high(t)
t += 6e-8
pb_1.go_low(t)
t+= 6e-8
stop(t)