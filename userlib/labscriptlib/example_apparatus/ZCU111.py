from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.ZCU111 import ZCU111, ZCU111DDS

PulseBlasterESRPro500(name= 'pb', loop_number = numIterations,  extra_flags = 0, extra_inst = 0, extra_inst_data = 0, extra_length= 0, inst_location = 0, additional_inst = 0, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")

#ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
DigitalOut('pb_1',pb.direct_outputs, 'flag 1')

DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

DigitalOut('pb_3',pb.direct_outputs, 'flag 3')
ZCU111(name = 'ZCU', parent_device = pb_clockline, com_port = 'COM5')
ZCU111DDS('ZCUDDS', ZCU, 'a')
 
t = 0 
add_time_marker(t, "Start", verbose = True)
start()

ZCUDDS.set_repetitions(t, '200000')
ZCUDDS.set_delay_time(t,'0')
ZCUDDS.set_start_src(t, 'external')
ZCUDDS.add_TTL(4, 0, 10e-9)
ZCUDDS.add_pulse(6,'buffer',0,2,32766,4000,0,'oneshot','product','[]')
'''ZCUDDS.add_pulse(0,'const',40e-9,20,32766,100,0,'oneshot','product','[]')
ZCUDDS.add_pulse(6,'buffer', 50e-9,2,32766,4000,90,'oneshot','product','[]')
ZCUDDS.add_pulse(6,'buffer', 54e-9,2,32766,4000,180,'oneshot','product','[]')
ZCUDDS.add_pulse(6,'buffer', 58e-9,2,32766,4000,270,'oneshot','product','[]')'''

pb_3.go_high(t)
t+= 1e-6

pb_3.go_low(t)
stop(t)
