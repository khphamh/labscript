from labscript import start, stop, add_time_marker, AnalogOut, DigitalOut
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock
from labscript_devices.ChaseAWG.labscript_devices import ChaseAWG, ChaseAWGDDS

# Use a virtual, or 'dummy', device for the psuedoclock
DummyPseudoclock(name='pseudoclock')

# An output of this DummyPseudoclock is its 'clockline' attribute, which we use
# to trigger children devices
ChaseAWG(name='chase', parent_device=pseudoclock.clockline)
ChaseAWGDDS('chaseDDS', chase, 'a')

t = 0
add_time_marker(t, "Start", verbose=True)
start()

chaseDDS.add_pulse(1, 256, 1, '2047.5 + 2047.5*sin( 2.0*pi* x/(64) )')
chaseDDS.add_pulse(1, 1024, 1, '2047.5 + 2047.5*sin( 2.0*pi* x/(128) )')
t += 1

# Stop the experiment shot with stop()
stop(t)
