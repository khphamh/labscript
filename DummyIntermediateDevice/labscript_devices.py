from labscript_devices import labscript_device, BLACS_tab, BLACS_worker
from labscript import IntermediateDevice, DigitalOut, AnalogOut, config
import numpy as np

class DummyIntermediateDevice(IntermediateDevice):

    description = 'Dummy IntermediateDevice'
    clock_limit = 1e6

    # If this is updated, then you need to update generate_code to support whatever types you add
    allowed_children = [DigitalOut, AnalogOut]

    def __init__(self, name, parent_device, BLACS_connection='dummy_connection', **kwargs):
        self.BLACS_connection = BLACS_connection
        IntermediateDevice.__init__(self, name, parent_device, **kwargs)

    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)
        group = self.init_device_group(hdf5_file)

        clockline = self.parent_device
        pseudoclock = clockline.parent_device
        times = pseudoclock.times[clockline]

        # out_table = np.empty((len(times),len(self.child_devices)), dtype=np.float32)
        # determine dtypes
        dtypes = []
        for device in self.child_devices:
            if isinstance(device, DigitalOut):
                device_dtype = np.int8
            elif isinstance(device, AnalogOut):
                device_dtype = np.float64
            dtypes.append((device.name, device_dtype))

        # create dataset
        out_table = np.zeros(len(times), dtype=dtypes)
        for device in self.child_devices:
            out_table[device.name][:] = device.raw_output

        group.create_dataset('OUTPUTS', compression=config.compression, data=out_table)