from labscript_devices import labscript_device, BLACS_tab, BLACS_worker
from labscript import IntermediateDevice, DigitalOut, AnalogOut, config
import numpy as np

class ChaseAWG(IntermediateDevice):

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




''' from labscript_devices import labscript_device, BLACS_tab, BLACS_worker
from labscript import IntermediateDevice, DigitalOut, AnalogOut, config, DDSQuantity, DDS
import numpy as np


class ChaseAWGDDS(DDSQuantity):
    description = 'ChaseAWGDDS'
    def __init__(self, *args, **kwargs):
        if 'call_parents_add_device' in kwargs:
            call_parents_add_device = kwargs['call_parents_add_device']
        else:
            call_parents_add_device = True

        kwargs['call_parents_add_device'] = False
        DDSQuantity.__init__(self, *args, **kwargs)

        self.gate = DigitalQuantity(self.name + '_gate', self, 'gate')
        self.phase_reset = DigitalQuantity(self.name + '_phase_reset', self, 'phase_reset')

        if call_parents_add_device:
            self.parent_device.add_device(self)

        self.pulse_list_1 = [ ]
        self.pulse_list_2 = [ ]
    def add_pulse(self,channel, NumPoints, NumLoops, Waveform):
        if channel == 1:
            self.pulse_list_1.append((NumPoints, NumLoops, Waveform))
        else:
            self.pulse_list_2.append((NumPoints, NumLoops, Waveform))

class ChaseAWG(IntermediateDevice):

    description = 'Dummy IntermediateDevice'
    clock_limit = 1e6

    # If this is updated, then you need to update generate_code to support whatever types you add
    allowed_children = [DigitalOut, AnalogOut]

    def __init__(self, name, parent_device, BLACS_connection='dummy_connection', **kwargs):
        self.BLACS_connection = BLACS_connection
        IntermediateDevice.__init__(self, name, parent_device, **kwargs)


    def _make_ChaseAWG_settings_table(self, inputs):
        """Collect analog input instructions and create the acquisition table"""
        if not inputs:
            return None

        for connection, input in inputs.items():
            pulse_list_1 = input.__dict__['pulse_list_1']
            pulse_list_2 = input.__dict__['pulse_list_2']

        settings = [(pulse_list_1, pulse_list_2)]
        settings_dtypes = [
            ('pulse_list_1', list),
            ('pulse_list_2', list)
        ]
        settings_table = np.empty(len(settings), dtype=settings_dtypes)
        for i, acq in enumerate(settings):
            settings_table[i] = acq
        return settings

    def generate_code(self, hdf5_file):
        IntermediateDevice.generate_code(self, hdf5_file)
        DDS_set = {}
        for device in self.child_devices:
            if isinstance(device, (DDS, ChaseAWGDDS)):
                DDS_set[device.connection] = device
        DDStable = self._make_ChaseAWG_settings_table(DDS_set)

        grp = self.init_device_group(hdf5_file)
        if DDStable is not None:
            grp.create_dataset('DDS', data=DDStable, compression=config.compression)'''