from labscript import *

#from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.ZCU4 import ZCU4, ZCU4DDS
from labscript_devices.SRS384 import SRS384, SRS384DDS
from labscript_devices.NI_DAQmx.models import NI_PCIe_6343

from labscript import *
from labscript_devices.PulseBlasterUSB import PulseBlasterUSB
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock

#from labscript_devices.NI_DAQmx.models import NI_PCIe_6343
ZCU4(name='zcu')
ClockLine(name = "pin0", pseudoclock=zcu.pseudoclock, connection = "flag 4")


#PulseBlasterUSB(name= 'pb',parent_device = zcu, trigger_device = pin0,  board_number = 0, programming_scheme = 'pb_start/BRANCH')
#ClockLine(name = "pb_clockline", pseudoclock = pb.pseudoclock, connection = "flag 3")


#SRS384(name = 'SRS', parent_device = pin0)


DigitalOut('Laser',zcu.direct_outputs, 'flag 1') #Laser switcxh

DigitalOut('MW_Switch',zcu.direct_outputs, 'flag 2') #MW switch
DigitalOut('a',zcu.direct_outputs, 'flag 3') #MW switch


#SRS384DDS('SRS1', SRS, 'a')

#ZCU4DDS('FPGADDS6', zcu.direct_outputs, '6')

#ZCU4DDS('FPGADDS5', zcu.direct_outputs, '5')


t = 0 
t_laser = t
t_MW = t
add_time_marker(t, "Start", verbose = True)
start()

SRS1.setamp(t, SRSAMP)
SRS1.setfreq(t, SRSFREQ)
SRS1.enable_output(t)
SRS1.IQ_enable(t)


"""Laser.go_high(t)
t+=10*(10**(-6))
Laser.go_low(t)
t+=100*(10**(-6))
Laser.go_high(t)
t+=10*(10**(-6))
Laser.go_low(t)
t+=100*(10**(-6))"""

Laser.go_high(t_laser)
t_laser += pol_duration
Laser.go_low(t_laser)
t_laser += charge_eq + tau
Laser.go_high(t_laser)
t_laser += readout_duration
Laser.go_low(t_laser)

t_MW += pol_duration + charge_eq
a.go_high(t_MW)
t_MW += pi2
a.go_low(t_MW)
t_MW += tau/2
a.go_high(t_MW)
t_MW += pi_length
a.go_low(t_MW)
t_MW += tau/2
a.go_high(t_MW)
t_MW += pi2
a.go_low(t_MW)

'''t_MW += pol_duration + charge_eq
MW_Switch.go_high(t_MW)
t_MW += pi2
MW_Switch.go_low(t_MW)
t_MW += tau/2
MW_Switch.go_high(t_MW)
t_MW += pi_length
MW_Switch.go_low(t_MW)
t_MW += tau/2
MW_Switch.go_high(t_MW)
t_MW += pi2
MW_Switch.go_low(t_MW)'''

t += 10**(-6) + max(t_laser, t_MW)


stop(t)
