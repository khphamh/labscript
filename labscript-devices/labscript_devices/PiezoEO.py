#####################################################################
#                                                                   #
# /PiezoEO.py                                                  #
#                                                                   #
# Copyright 2013, Monash University                                 #
#                                                                   #
# This file is part of the module labscript_devices, in the         #
# labscript suite (see http://labscriptsuite.org), and is           #
# licensed under the Simplified BSD License. See the license.txt    #
# file in the root of the project for the full license.             #
#                                                                   #
#####################################################################
from msilib import sequence
from labscript_devices import runviewer_parser, labscript_device, BLACS_tab, BLACS_worker

from labscript import Device, IntermediateDevice, Pseudoclock, ClockLine, PseudoclockDevice, config, LabscriptError, StaticAnalogQuantity, AnalogOut, DigitalOut, set_passed_properties, WaitMonitor, compiler, DDS, DDSQuantity, DigitalQuantity
import copy

import numpy as np
import labscript_utils.h5_lock, h5py
from ctypes import *
import struct
import serial
import time
max_deviations = [
    {'low': 0.1,        'high': 249.999,    'dev': 10.0},
    {'low': 249.999,    'high': 500.0,      'dev': 5.0},
    {'low': 500.0,      'high': 1000.0,     'dev': 10.0},
    {'low': 1000.0,     'high': 2000.0,     'dev': 20.0},
    {'low': 2000.0,     'high': 4000.0,     'dev': 40.0}
]


class PiezoEODDS(DDSQuantity):
    description = 'PiezoEODDS'
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


class PiezoEO(IntermediateDevice):
    # This device can only have Pseudoclock children (digital outs and DDS outputs should be connected to a child device)
    allowed_children = [DigitalOut, DDS, PiezoEODDS]

    def __init__(self, name, parent_device=None, clock_terminal=None,  **kwargs):
        self.BLACS_connection = None
        # Create the internal pseudoclock
        # Create the internal direct output clock_line
        IntermediateDevice.__init__(self, name, parent_device, **kwargs)

    def add_device(self, device):
        IntermediateDevice.add_device(self, device)


    def _check_wait_monitor_ok(self):
        if (
            compiler.master_pseudoclock is self
            and compiler.wait_table
            and compiler.wait_monitor is None
            and self.programming_scheme != 'pb_stop_programming/STOP'
        ):
            msg = """If using waits without a wait monitor, the PiezoEO used as a
                master pseudoclock must have
                programming_scheme='pb_stop_programming/STOP'. Otherwise there is no way
                for BLACS to distinguish between a wait, and the end of a shot. Either
                use a wait monitor (see labscript.WaitMonitor for details) or set
                programming_scheme='pb_stop_programming/STOP for %s."""
            raise LabscriptError(dedent(msg) % self.name)


    def _make_PiezoEO_settings_table(self, inputs):
        """Collect analog input instructions and create the acquisition table"""
        if not inputs:
            return None

        for connection, input in inputs.items():
            amp = input.__dict__['amplitude'].__dict__['raw_output'][0]


        settings = [(amp)]
        settings_dtypes = [('amp', float)]
        settings_table = np.empty(len(settings), dtype=settings_dtypes)
        for i, acq in enumerate(settings):
            settings_table[i] = acq
        return settings
    def generate_code(self, hdf5_file):
        # Generate the hardware instructions
        IntermediateDevice.generate_code(self, hdf5_file)
        DDS_set = {}
        for device in self.child_devices:
            if isinstance(device, (DDS, PiezoEODDS)):
                DDS_set[device.connection] = device
        DDStable = self._make_PiezoEO_settings_table(DDS_set)

        grp = self.init_device_group(hdf5_file)
        if DDStable is not None:
            grp.create_dataset('DDS', data=DDStable, compression=config.compression)

class PiezoEODirectOutputs(IntermediateDevice):
    allowed_children = [DDS, PiezoEODDS, DigitalOut]
    description = 'PB-DDSII-300 Direct Outputs'
  
    def add_device(self, device):
        IntermediateDevice.add_device(self, device)
        if isinstance(device, DDS):
            # Check that the user has not specified another digital line as the gate for this DDS, that doesn't make sense.
            # Then instantiate a DigitalQuantity to keep track of gating.
            if device.gate is None:
                device.gate = DigitalQuantity(device.name + '_gate', device, 'gate')
            else:
                raise LabscriptError('You cannot specify a digital gate ' +
                                     'for a DDS connected to %s. '% (self.name) + 
                                     'The digital gate is always internal to the ZCU4.')
import time

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab
from qtutils.qt import QtWidgets

@BLACS_tab
class PiezoEOTab(DeviceTab):
    def initialise_GUI(self):
        # Capabilities
                # Create status labels

        self.base_units =    {'amp':'um'}
        self.base_min =      {'amp':0}
        self.base_max =      {'amp':100}
        self.base_step =     {'amp':0.1}
        self.base_decimals = {'amp':4} # TODO: find out what the phase precision is!

        # Create DDS Output objects
        RF_prop = {}
        
        RF_prop['Output Channel'] = {}
        for subchnl in ['amp']:
            RF_prop['Output Channel'][subchnl] = {'base_unit':self.base_units[subchnl],
                                                'min':self.base_min[subchnl],
                                                'max':self.base_max[subchnl],
                                                'step':self.base_step[subchnl],
                                                'decimals':self.base_decimals[subchnl]
                                                }
        
        # Create the output objects
        self.create_dds_outputs(RF_prop)

        # Create widgets for output objects
        dds_widgets,ao_widgets,do_widgets = self.auto_create_widgets()
        # and auto place the widgets in the UI
        self.auto_place_widgets(("RF Output",dds_widgets) )


        # Create and set the primary worker
        self.create_worker("main_worker", PiezoEOWorker)
        self.primary_worker = "main_worker"

        # Create status labels
        self.status_label = QtWidgets.QLabel("Status: Unknown")
        self.clock_status_label = QtWidgets.QLabel("Clock status: Unknown")
        self.get_tab_layout().addWidget(self.status_label)
        self.get_tab_layout().addWidget(self.clock_status_label)

        # Set the capabilities of this device
        self.supports_smart_programming(True)

        # Create status monitor timout
        self.statemachine_timeout_add(2000, self.status_monitor)
        # Set the capabilities of this device
        self.supports_remote_value_check(False) # !!!
        self.supports_smart_programming(False) # !!!
    @define_state(
        MODE_MANUAL
        | MODE_BUFFERED
        | MODE_TRANSITION_TO_BUFFERED
        | MODE_TRANSITION_TO_MANUAL,
        True,
    )
    def status_monitor(self, notify_queue=None):
        """Gets the status of the PrawnBlaster from the worker.

        When called with a queue, this function writes to the queue
        when the PrawnBlaster is waiting. This indicates the end of
        an experimental run.

        Args:
            notify_queue (:class:`~queue.Queue`): Queue to notify when
                the experiment is done.

        """

        status, clock_status, waits_pending = yield (
            self.queue_work(self.primary_worker, "check_status")
        )

        # Manual mode or aborted
        done_condition = status == 0 or status == 5
        done_condition = True
        # Update GUI status/clock status widgets
        self.status_label.setText(f"Status: {status}")
        self.clock_status_label.setText(f"Clock status: {clock_status}")

        if notify_queue is not None and done_condition and not waits_pending:
            # Experiment is over. Tell the queue manager about it, then
            # set the status checking timeout back to every 2 seconds
            # with no queue.
            notify_queue.put("done")
            self.statemachine_timeout_remove(self.status_monitor)
            self.statemachine_timeout_add(2000, self.status_monitor)


import re
class PiezoEOWorker(Worker):

    def init(self):

        global h5py; import labscript_utils.h5_lock, h5py
        self.mydll = cdll.LoadLibrary("C:\Program Files\Edmund Optics\EO-Drive\EO-Drive.dll")
        functions_list = ['EO_GetHandleBySerial', 'EO_InitHandle', 'EO_InitAllHandles','EO_GetAllHandles','EO_NumberOfCurrentHandles','EO_ReleaseHandle','EO_ReleaseAllHandles','EO_Move','EO_GetMaxCommand','EO_GetCommandPosition','EO_GetSerialNumber']
        for i in functions_list:
            exec('self.mydll.' + i)
        self.mydll.EO_ReleaseAllHandles()
        self.handle = self.mydll.EO_InitHandle()
        self.logger.info("current handle is " + str(self.handle))
        #self.mydll.EO_Move.argtype = (c_int, c_double)
    def check_status(self):
        return 2, 0, False

    def check_remote_values(self):
        results = {}


        return results


    def program_manual(self,front_panel_values):
        results = {}

        results['Output Channel']=  {}

        values = front_panel_values['Output Channel']
        v_amp = values['amp']
        results['Output Channel']['amp'] = v_amp
        #raise LabscriptError(v_amp)
        success_value = self.mydll.EO_Move(self.handle, int(v_amp))
        self.logger.info("attempted move:" + str(success_value))
        return results

    def start_run(self):

        self.started = True

    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):
        self.h5file = h5file
        self.started = False
        return_values = {'a': 1}

        with h5py.File(h5file,'r') as hdf5_file:
            group = hdf5_file['devices/%s'%device_name]
            DDS_table = group['DDS'][:]
            v_amp = DDS_table[0]
            self.mydll.EO_Move(self.handle,int(v_amp))

        return return_values

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        return self.transition_to_manual(True)

    def transition_to_manual(self,abort = False):


        return True

    def shutdown(self):

        self.mydll.EO_ReleaseAllHandles()
        return

import labscript_utils.h5_lock  # noqa: F401
import h5py
import numpy as np

import labscript_utils.properties as properties


class PiezoEOParser(object):
    """Runviewer parser for the PiezoEO Pseudoclocks."""
    def __init__(self, path, device):
        """
        Args:
            path (str): path to h5 shot file
            device (str): labscript name of PiezoEO device
        """
        self.path = path
        self.name = device.name
        self.device = device

    def get_traces(self, add_trace, clock=None):
        """Reads the shot file and extracts hardware instructions to produce
        runviewer traces.

        Args:
            add_trace (func): function handle that adds traces to runviewer
            clock (tuple, optional): clock times from timing device, if not
                the primary pseudoclock

        Returns:
            dict: Dictionary of clocklines and triggers derived from instructions
        """

        if clock is not None:
            times, clock_value = clock[0], clock[1]
            clock_indices = np.where((clock_value[1:] - clock_value[:-1]) == 1)[0] + 1
            # If initial clock value is 1, then this counts as a rising edge
            # (clock should be 0 before experiment) but this is not picked up
            # by the above code. So we insert it!
            if clock_value[0] == 1:
                clock_indices = np.insert(clock_indices, 0, 0)
            clock_ticks = times[clock_indices]

        # get the pulse program
        pulse_programs = []
        with h5py.File(self.path, "r") as f:
            # Get the device properties
            device_props = properties.get(f, self.name, "device_properties")
            conn_props = properties.get(f, self.name, "connection_table_properties")

            self.clock_resolution = device_props["clock_resolution"]
            self.trigger_delay = device_props["trigger_delay"]
            self.wait_delay = device_props["wait_delay"]

            # Extract the pulse programs
            num_pseudoclocks = conn_props["num_pseudoclocks"]
            for i in range(num_pseudoclocks):
                pulse_programs.append(f[f"devices/{self.name}/PULSE_PROGRAM_{i}"][:])

        # Generate clocklines and triggers
        clocklines_and_triggers = {}
        
        for pseudoclock_name, pseudoclock in self.device.child_list.items():
            # Get pseudoclock index
            connection_parts = pseudoclock.parent_port.split()
            # Skip if not one of the 4 possible pseudoclock outputs (there is one for
            # the wait monitor too potentially)
            if connection_parts[0] != "pseudoclock":
                continue

            # Get the pulse program
            index = int(connection_parts[1])
            pulse_program = pulse_programs[index]

            time = []
            states = []
            trigger_index = 0
            t = 0 if clock is None else clock_ticks[trigger_index] + self.trigger_delay
            trigger_index += 1

            clock_factor = self.clock_resolution / 2.0

            last_instruction_was_wait = False
            for row in pulse_program:
                if row["reps"] == 0 and not last_instruction_was_wait:  # WAIT
                    last_instruction_was_wait = True
                    if clock is not None:
                        t = clock_ticks[trigger_index] + self.trigger_delay
                        trigger_index += 1
                    else:
                        t += self.wait_delay
                elif last_instruction_was_wait:
                    # two waits in a row means an indefinite wait, so we just skip this
                    # instruction.
                    last_instruction_was_wait = False
                    continue
                else:
                    last_instruction_was_wait = False
                    for i in range(row["reps"]):
                        for j in range(1, -1, -1):
                            time.append(t)
                            states.append(j)
                            t += row["half_period"] * clock_factor

            pseudoclock_clock = (np.array(time), np.array(states))

            for clock_line_name, clock_line in pseudoclock.child_list.items():
                # Ignore the dummy internal wait monitor clockline
                if clock_line.parent_port.startswith("GPIO"):
                    clocklines_and_triggers[clock_line_name] = pseudoclock_clock
                    add_trace(
                        clock_line_name, pseudoclock_clock, self.name, clock_line.parent_port
                    )

        return clocklines_and_triggers



import labscript_devices

labscript_device_name = 'PiezoEO'
blacs_tab = 'labscript_devices.PiezoEO.PiezoEOTab'
parser = 'labscript_devices.PiezoEO.PiezoEOParser'

labscript_devices.register_classes(
    labscript_device_name=labscript_device_name,
    BLACS_tab=blacs_tab,
    runviewer_parser=parser,
)