from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
PulseBlasterESRPro500(name= 'pb', loop_number = numIterations, board_number = 0, programming_scheme = 'pb_start/BRANCH')
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
CounterIn("counter", Dev1, connection = "ctr2", CPT_connection = "PFI13", trigger = "PFI4", numIterations = numIterations)
#ctr 1 is cpt

DigitalOut('daq_dout_8', Dev1, 'port0/line8') 
DigitalOut('daq_dout_9', Dev1, 'port0/line9') 

t = 0  
add_time_marker(t, "Start", verbose = True)
start()
dt = 2e-3
#pb_1.go_high(t)
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
    #pb_1.go_high(t)
    #pb_1.go_low(t)
#pb_1.go_low(t)

stop(t)

















"""for i in range(x_points):
    anaout_0.constant(t, V_x_offset+V_x_min + i*(V_x_max-V_x_min)/x_points)
    for j in range(y_points):
        anaout_1.constant(t, V_y_offset + V_y_min + j*(V_y_max-V_y_min)/y_points)
        t+=dt        
        daq_dout_8.go_high(t)
        counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = t, end_time =t+2*dt, sample_freq = 1e5)
        t+=dt        
        t+=dt        
        daq_dout_8.go_low(t)     
    for j in range(y_points):
        anaout_1.constant(t, V_y_offset + V_y_max - j*(V_y_max-V_y_min)/y_points)
        t+=dt        
        daq_dout_8.go_high(t)
        counter.acquire(label = 'count_'+str(i)+'_'+str(y_points - 1 - j), start_time = t, end_time =t+2*dt, sample_freq = 1e5)
        t+=dt        
        t+=dt        
        daq_dout_8.go_low(t) """    

"""for i in range(x_points):
    anaout_0.constant(t, V_x_offset+V_x_min + i*(V_x_max-V_x_min)/x_points)
    for j in range(y_points):
        anaout_1.constant(t, V_y_offset + V_y_min + j*(V_y_max-V_y_min)/y_points)
        t+=dt        
        daq_dout_8.go_high(t)
        counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = t, end_time =t+3*dt, sample_freq = 1e5)
        t+=3*dt        
        daq_dout_8.go_low(t)     
    for j in range(y_points):
        anaout_1.constant(t, V_y_offset + V_y_min + j*(V_y_max-V_y_min)/y_points)
        t+=dt        
        daq_dout_8.go_high(t)
        counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = t, end_time =t+3*dt, sample_freq = 1e5)
        t+=3*dt        
        daq_dout_8.go_low(t)"""
