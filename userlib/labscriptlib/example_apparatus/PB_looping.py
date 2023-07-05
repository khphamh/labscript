from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

PulseBlasterESRPro500(name= 'pb', loop_number = numIterations,  extra_flags = 0, extra_inst = 0, extra_inst_data = 0, extra_length= 0, inst_location = 0, additional_inst = 0, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")


DigitalOut('pb_1',pb.direct_outputs, 'flag 1')
DigitalOut('pb_2',pb.direct_outputs, 'flag 2')


t = 0  
add_time_marker(t, "Start", verbose = True)
start()
'''t+= 20e-9
for i in range(10):
    pb_1.go_high(t)
    t += dt
    pb_1.go_low(t)
    t += dt'''
'''pb_2.go_high(t)
t+= dt'''
#t += 20e-9
#pb_2.go_high(t)
#for i in range(19):
'''pb_1.go_high(t)
t += 0.5e-5
pb_1.go_low(t)
t+= 0.5e-5
t+= 1e-3
pb_1.go_high(t)
t += 1e-5
pb_1.go_low(t)
t+= 1e-5
t+= 20e-9
pb_1.go_high(t)
t += 6e-8
pb_1.go_low(t)
t+= 6e-8'''
'''pb_2.go_low(t)
t += 100e-6
pb_2.go_high(t)
for i in range(10):
    pb_1.go_high(t)
    t += 0.5e-4
    pb_1.go_low(t)
    t+= 0.5e-4
pb_2.go_low(t)'''
#pb_2.go_high(t)
#t+= 2e-8
for i in range(200000):
    t += 20e-9
    #pb_2.go_high(t)
    #t += 10e-6
    pb_1.go_high(t)
    t += 0.5e-5 
    pb_1.go_low(t)
    #pb_2.go_low(t)
    '''t += 12e-9

    t+= 60e-9
    t+=0.5e-4
    t+= 12e-9'''
    #pb_2.go_high(t)
    t+=0.5e-5
    pb_1.go_high(t)
    t += 1e-5
    pb_1.go_low(t)
    #pb_2.go_low(t)
    t+= 1e-5
pb_1.go_high(t)
t += 6e-8
pb_1.go_low(t)
t+= 6e-8
pb_1.go_high(t)
t += 6e-8
pb_1.go_low(t)
t+= 6e-8
stop(t)
