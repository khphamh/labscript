from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.ZCU111 import ZCU111, ZCU111DDS

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
PulseBlasterESRPro500(name= 'pb', loop_number = numIterations,  extra_flags = 0, extra_inst = 3, extra_inst_data = 2, extra_length= 10, inst_location = 10, additional_inst = 8, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")
ClockLine(name = "pb_clockline_2", pseudoclock = pb.pseudoclock, connection = "flag 4")

NI_PCIe_6343(name = 'Dev2',
    parent_device = pb_clockline,
    clock_terminal = '/Dev2/PFI5',
    MAX_name = 'Dev2',
    stop_order = -1,
    acquisition_rate = 1e5
    )
DigitalOut('pb_1',pb.direct_outputs, 'flag 1')
DigitalOut('pb_2',pb.direct_outputs, 'flag 2')
DigitalOut('pb_3',pb.direct_outputs, 'flag 3')

AnalogOut('anaout_0', Dev2, 'ao0')
AnalogOut('anaout_1', Dev2, 'ao1')
AnalogOut('anaout_2', Dev2, 'ao2')
AnalogOut('anaout_3', Dev2, 'ao3')
CounterIn("counter", Dev2, connection = "ctr2", CPT_connection = "PFI0", trigger = "PFI1", numIterations = numIterations)
#ctr 1 is cpt

DigitalOut('daq_dout_8', Dev2, 'port0/line8') 
DigitalOut('daq_dout_9', Dev2, 'port0/line9') 
ZCU111(name = 'ZCU', parent_device = pb_clockline_2, com_port = 'COM5')
ZCU111DDS('ZCUDDS', ZCU, 'a')

t = 0  
add_time_marker(t, "Start", verbose = True)
start()
#dt = 5e-3
counter_gate_time = 400e-9
counter.fast_counter(1e8)
ZCUDDS.set_repetitions(t, '10')
ZCUDDS.set_delay_time(t,'16.5') #us
ZCUDDS.set_start_src(t, 'external')
ZCUDDS.add_TTL(4, 0, 10e-9)
ZCUDDS.add_pulse(6,'const',0,pulse_time,32766,4000,0,'oneshot','product','[]')
#for i in range(10):
t += 20e-9
pb_2.go_high(t)
t += 10e-6
#counter.acquire(label = 'count', start_time = t, end_time =t+counter_gate_time, sample_freq = 1e8)
pb_1.go_high(t)
t += counter_gate_time
pb_1.go_low(t)
pb_2.go_low(t)
pb_3.go_high(t)
t+= 20e-9
pb_3.go_low(t)
t+=5e-6
'''t += 12e-9

t+= 60e-9
#t+=0.5e-4
t+= 12e-9'''
pb_2.go_high(t)
pb_1.go_high(t)
t += counter_gate_time
pb_1.go_low(t)
pb_2.go_low(t)
t+= 1e-6
'''pb_1.go_high(t)
t += 1e-5
pb_1.go_low(t)
#pb_2.go_low(t)
t+= 1e-5'''

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