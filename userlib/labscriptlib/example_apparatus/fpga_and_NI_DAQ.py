from labscript import *

#from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.ZCU4 import ZCU4, ZCU4DDS, ZCU4TTL
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343


ZCU4(name='pb', com_port = 'COM5')

#ClockLine('pb_cl', pb.pseudoclock, 'flag 0')
ZCU4DDS('DDS6', pb.direct_outputs, '6')
ZCU4TTL('TTL', pb.direct_outputs, 't')

ClockLine(name = "pin0", pseudoclock = pb.pseudoclock, connection = "flag 0")
NI_PCIe_6343(name = 'Dev2',
    parent_device = pin0,
    clock_terminal = '/Dev2/PFI15',
    MAX_name = 'Dev2',
    stop_order = -1,
    acquisition_rate = 1e5
    )


AnalogOut('anaout_0', Dev2, 'ao0')
AnalogOut('anaout_1', Dev2, 'ao1')
AnalogOut('anaout_2', Dev2, 'ao2')
AnalogOut('anaout_3', Dev2, 'ao3')
CounterIn("counter", Dev2, connection = "ctr2", CPT_connection = "PFI13", trigger = "PFI4", numIterations = numIterations)
#ctr 1 is cpt

DigitalOut('daq_dout_8', Dev2, 'port0/line8') 
DigitalOut('daq_dout_9', Dev2, 'port0/line9') 
t = 0 
s=0
add_time_marker(t, "Start", verbose = True)
start()
DDS6.add_pulse(0, 'const', t, 10, 3000, 100, 0, 'oneshot', 'product', '[]')

dt = 2e-4
t += 2
for i in range(x_points):
    if i % 2:
        for j in range(y_points):
            TTL.add_TTL(1, t, t + 5*(10**(-9)))
            t+=dt        
            TTL.add_TTL(1, t, t + 5*(10**(-9)))

            t+=3*dt
            TTL.add_TTL(1, t, t + 5*(10**(-9)))
    else:
        for j in range(y_points):
            TTL.add_TTL(1, t, t + 5*(10**(-9)))
            t+=dt
            TTL.add_TTL(1, t, t + 5*(10**(-9)))

            t+=3*dt        
            TTL.add_TTL(1, t, t + 5*(10**(-9)))
    t+=dt
for i in range(x_points):
    anaout_0.constant(s, V_x_offset+V_x_min + i*(V_x_max-V_x_min)/x_points)
    if i % 2:
        for j in range(y_points):
            anaout_1.constant(s, V_y_offset + V_y_min + j*(V_y_max-V_y_min)/y_points)
            s+=dt        
            daq_dout_8.go_high(s)
            counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = s, end_time =s+3*dt, sample_freq = 1e5)

            s+=3*dt
            daq_dout_8.go_low(s)
    else:
        for j in range(y_points):
            anaout_1.constant(s, V_y_offset + V_y_max - j*(V_y_max-V_y_min)/y_points)
            s+=dt
            daq_dout_8.go_high(s)
            counter.acquire(label = 'count_'+str(i)+'_'+str(j), start_time = s, end_time =s+3*dt, sample_freq = 1e5)

            s+=3*dt        
            daq_dout_8.go_low(s)
    s +=dt
stop(t)
