from labscript import *
from labscript_devices.PulseBlasterUSB import PulseBlasterUSB
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock
from labscript_devices.SRS384 import SRS384, SRS384DDS

from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
PulseBlasterUSB(name= 'pb', loop_number = numIterations, board_number = 0, programming_scheme = 'pb_start/BRANCH')
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
SRS384(name = 'SRS', parent_device = pb_clockline, com_port = 'COM9')
SRS384DDS('SRSDDS', SRS, 'a')

t = 0  
add_time_marker(t, "Start", verbose = True)
start()
SRSDDS.setamp(t, -35)
SRSDDS.setfreq(t, 3500)
SRSDDS.enable_mod(t)
SRSDDS.enable_freq_sweep(t)
SRSDDS.set_sweep_rate(t, 10)
SRSDDS.set_sweep_dev(t, 500e6)

pb_1.go_high(t)
pb_2.go_high(t)
dt = 10e-5
for i in range(100):
    anaout_2.ramp(t=t, duration = 2*200*dt, initial = -1, final = 1, samplerate = 1/(2*dt))
    for j in range(200):
        t += dt
        daq_dout_8.go_high(t)
        counter.acquire(label = 'count_up'+str(j), start_time = t, end_time =t + dt, sample_freq = 1e5)
        t+= dt
        daq_dout_8.go_low(t)

    anaout_2.ramp(t=t, duration = 2*200*dt, initial = 1, final = -1, samplerate = 1/(2*dt))
    for j in range(200):
        t += dt
        daq_dout_8.go_high(t)
        counter.acquire(label = 'count_down'+str(200-j), start_time = t, end_time =t + dt, sample_freq = 1e5)
        t+= dt
        daq_dout_8.go_low(t)
stop(t)

'''anaout_2.ramp(t=t, duration = 4*dt, initial = -1, final = 1, samplerate = 1/(201*4*dt))
for j in range(201):
    t += dt
    daq_dout_8.go_high(t)
    counter.acquire(label = 'count_up'+str(j), start_time = t, end_time =t + 3*dt, sample_freq = 1e5)
    t+= dt
    pb_2.go_high(t)
    t += dt
    pb_2.go_low(t)
    t += dt
    daq_dout_8.go_low(t)

anaout_2.ramp(t=t, duration = 4*dt, initial = 1, final = -1, samplerate = 1/(201*4*dt))
for j in range(201):
    t += dt
    daq_dout_8.go_high(t)
    counter.acquire(label = 'count_down'+str(201-j), start_time = t, end_time =t + 3*dt, sample_freq = 1e5)
    t+= dt
    pb_2.go_high(t)
    t += dt
    pb_2.go_low(t)
    t += dt
    daq_dout_8.go_low(t)
pb_1.go_low(t)'''