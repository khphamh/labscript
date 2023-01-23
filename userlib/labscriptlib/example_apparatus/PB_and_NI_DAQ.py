from labscript import *
from labscript_devices.PulseBlasterUSB import PulseBlasterUSB
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

PulseBlasterUSB(name= 'pb', loop_number = numIterations, board_number = 0, programming_scheme = 'pb_start/BRANCH')
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

for iter in range(numIterations):
    dt = 100e-6
    daq_dout_8.go_high(t)
    counter.acquire(label = 'ref1_'+str(iter), start_time = t, end_time =t+dt, sample_freq = 1e6)
    t+=dt
    daq_dout_8.go_low(t)
    t+=700e-6

    anaout_0.constant(t, 0)
    anaout_1.constant(t, 0)
    # for i in range(10):
    #     for j in range(10):
    #         anaout_1.constant(t, -1 + j*0.02)
    #         t+=5*0.5*(10**(-2))
    #     anaout_0.constant(t, -1 + i*0.02)
    # daq_dout_8.go_low(t)
    t+= 10e-6
    daq_dout_8.go_high(t)
    counter.acquire(label = 'bright_'+str(iter), start_time = t, end_time =t+dt, sample_freq = 1e6)
    t+=dt
    daq_dout_8.go_low(t)
    t+=700e-6
    daq_dout_8.go_high(t)
    counter.acquire(label = 'ref2_'+str(iter), start_time = t, end_time =t+dt, sample_freq = 1e6)
    t+=dt
    daq_dout_8.go_low(t)
    t+=700e-6

    daq_dout_8.go_high(t)
    counter.acquire(label = 'dark_'+str(iter), start_time = t, end_time =t+dt, sample_freq = 1e6)
    t+=dt
    daq_dout_8.go_low(t)
    t+=700e-6

t += 1e-6
stop(t)
