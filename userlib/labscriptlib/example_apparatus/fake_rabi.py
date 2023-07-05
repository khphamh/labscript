from labscript import *
from labscript_devices.PulseBlasterESRPro500 import PulseBlasterESRPro500
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343


PulseBlasterESRPro500(name= 'pb', loop_number = 2*numIterations,  extra_flags = 0, extra_inst = 3, extra_inst_data = 2, extra_length= 10, inst_location = 7, additional_inst = 8, board_number = 0, programming_scheme = 'pb_start/BRANCH')
ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 0")
ClockLine(name = "pb_clockline_5", pseudoclock = pb.pseudoclock, connection = "flag 5")

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
DigitalOut('pb_4',pb.direct_outputs, 'flag 4') #open mw switch
DigitalOut('pb_6',pb.direct_outputs, 'flag 6') #dummy

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
counter.fast_counter(1e8,2)
t += 20e-9
anaout_0.constant(t, V_x_offset)
anaout_1.constant(t, V_y_offset)
pb_2.go_high(t)
pb_4.go_high(t)
for i in range(repetitions):
    for j in range(N_data_points):
        #anaout_2.constant(t, -1 + 2*j/(N_data_points - 1)) 
        t+=dt
        pb_1.go_high(t)
        t+=3*dt        
        pb_1.go_low(t)        
        
    for j in range(N_data_points):
        #anaout_2.constant(t, 1 - 2*j/(N_data_points - 1)) 
        t+=dt
        pb_1.go_high(t)
        t+=3*dt        
        pb_1.go_low(t)        
            
pb_2.go_low(t)
pb_4.go_low(t)
t += 10e-6
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

