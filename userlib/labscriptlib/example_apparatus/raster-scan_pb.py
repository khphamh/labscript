from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
PulseBlasterESRPro500(name= 'pb', loop_number = numIterations, extra_flags = 0, extra_inst = 0, extra_inst_data = 0, extra_length= 0, inst_location = 0, additional_inst = 0, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")
NI_PCIe_6343(name = 'Dev2',
    parent_device = pb_clockline,
    clock_terminal = '/Dev2/PFI5',
    MAX_name = 'Dev2',
    stop_order = -1,
    acquisition_rate = 1e5
    )
DigitalOut('pb_1',pb.direct_outputs, 'flag 1')
DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

AnalogOut('anaout_0', Dev2, 'ao0')
AnalogOut('anaout_1', Dev2, 'ao1')
AnalogOut('anaout_2', Dev2, 'ao2')
AnalogOut('anaout_3', Dev2, 'ao3')
CounterIn("counter", Dev2, connection = "ctr2", CPT_connection = "PFI13", trigger = "PFI4", numIterations = numIterations)
#ctr 1 is cpt

DigitalOut('daq_dout_8', Dev2, 'port0/line8') 
DigitalOut('daq_dout_9', Dev2, 'port0/line9') 

t = 0  
add_time_marker(t, "Start", verbose = True)
start()
#dt = 5e-3
t += 20e-9
pb_2.go_high(t)
for i in range(x_points):
    anaout_0.constant(t, V_x_offset+V_x_min + i*(V_x_max-V_x_min)/x_points)
    if i % 2:
        for j in range(y_points):
            anaout_1.constant(t, V_y_offset + V_y_min + j*(V_y_max-V_y_min)/y_points)
            t+=dt        
            daq_dout_8.go_high(t)
            counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = t, end_time =t+3*dt, sample_freq = 1e5)

            t+=3*dt        
            daq_dout_8.go_low(t)     
    else:
        for j in range(y_points):
            anaout_1.constant(t, V_y_offset + V_y_max - j*(V_y_max-V_y_min)/y_points)
            t+=dt        
            daq_dout_8.go_high(t)
            counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = t, end_time =t+3*dt, sample_freq = 1e5)
            t+=3*dt        
            daq_dout_8.go_low(t)   
pb_2.go_low(t)
 
stop(t)

'''
from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
PulseBlasterESRPro500(name= 'pb', loop_number = numIterations, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")
NI_PCIe_6343(name = 'Dev2',
    parent_device = pb_clockline,
    clock_terminal = '/Dev2/PFI5',
    MAX_name = 'Dev2',
    stop_order = -1,
    acquisition_rate = 1e5
    )
DigitalOut('pb_1',pb.direct_outputs, 'flag 1')
DigitalOut('pb_2',pb.direct_outputs, 'flag 2')

AnalogOut('anaout_0', Dev2, 'ao0')
AnalogOut('anaout_1', Dev2, 'ao1')
AnalogOut('anaout_2', Dev2, 'ao2')
AnalogOut('anaout_3', Dev2, 'ao3')
CounterIn("counter", Dev2, connection = "ctr2", CPT_connection = "PFI13", trigger = "PFI4", numIterations = numIterations)
#ctr 1 is cpt

DigitalOut('daq_dout_8', Dev2, 'port0/line8') 
DigitalOut('daq_dout_9', Dev2, 'port0/line9') 

t = 0  
add_time_marker(t, "Start", verbose = True)
start()
#dt = 5e-3
pb_2.go_high(t)
for i in range(x_points):
    anaout_0.constant(t, V_x_offset+V_x_min + i*(V_x_max-V_x_min)/x_points)
    if i % 2:
        for j in range(y_points):
            anaout_1.constant(t, V_y_offset + V_y_min + j*(V_y_max-V_y_min)/y_points)
            t+=dt        
            #daq_dout_8.go_high(t)
            counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = t, end_time =t+3*dt, sample_freq = 1e8)
            s = t
            for k in range(6):
                pb_1.go_high(s)
                s += 5e-8
                pb_1.go_low(s)
                s += 5e-8

            t+=3*dt        
            #daq_dout_8.go_low(t)     
    else:
        for j in range(y_points):
            anaout_1.constant(t, V_y_offset + V_y_max - j*(V_y_max-V_y_min)/y_points)
            t+=dt        
            #daq_dout_8.go_high(t)
            counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = t, end_time =t+3*dt, sample_freq = 1e8)
            s = t
            for k in range(6):
                pb_1.go_high(s)
                s += 5e-8
                pb_1.go_low(s)
                s += 5e-8
            t+=3*dt        
            #daq_dout_8.go_low(t)   
pb_2.go_low(t)
 
stop(t)
'''