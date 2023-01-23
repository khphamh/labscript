#####################################################################
#                                                                   #
# /NI_DAQmx/blacs_workers.py                                        #
#                                                                   #
# Copyright 2018, Monash University, JQI, Christopher Billington    #
#                                                                   #
# This file is part of the module labscript_devices, in the         #
# labscript suite (see http://labscriptsuite.org), and is           #
# licensed under the Simplified BSD License. See the license.txt    #
# file in the root of the project for the full license.             #
#                                                                   #
#####################################################################
import sys
import time
import threading
from PyDAQmx import *
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxTypes import *
from PyDAQmx.DAQmxCallBack import *

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import labscript_utils.h5_lock
import h5py
from zprocess import Event
from zprocess.utils import _reraise

import labscript_utils.properties as properties
from labscript_utils import dedent
from labscript_utils.connections import _ensure_str

from blacs.tab_base_classes import Worker
from labscript import LabscriptError

from .utils import split_conn_port, split_conn_DO, split_conn_AI
from .daqmx_utils import incomplete_sample_detection


class NI_DAQmxOutputWorker(Worker):
    def init(self):
        self.check_version()
        # Reset Device: clears previously added routes etc. Note: is insufficient for
        # some devices, which require power cycling to truly reset.
        DAQmxResetDevice(self.MAX_name)
        self.start_manual_mode_tasks()

    def stop_tasks(self):
        if self.AO_task is not None:
            self.AO_task.StopTask()
            self.AO_task.ClearTask()
            self.AO_task = None
        if self.DO_task is not None:
            self.DO_task.StopTask()
            self.DO_task.ClearTask()
            self.DO_task = None

    def shutdown(self):
        self.stop_tasks()

    def check_version(self):
        """Check the version of PyDAQmx is high enough to avoid a known bug"""
        major = uInt32()
        minor = uInt32()
        patch = uInt32()
        DAQmxGetSysNIDAQMajorVersion(major)
        DAQmxGetSysNIDAQMinorVersion(minor)
        DAQmxGetSysNIDAQUpdateVersion(patch)

        if major.value == 14 and minor.value < 2:
            msg = """There is a known bug with buffered shots using NI DAQmx v14.0.0.
                This bug does not exist on v14.2.0. You are currently using v%d.%d.%d.
                Please ensure you upgrade to v14.2.0 or higher."""
            raise Exception(dedent(msg) % (major.value, minor.value, patch.value))

    def start_manual_mode_tasks(self):
        # Create tasks:
        if self.num_AO > 0:
            self.AO_task = Task()
        else:
            self.AO_task = None

        if self.ports:
            self.DO_task = Task()
        else:
            self.DO_task = None

        # Setup AO channels
        for i in range(self.num_AO):
            con = self.MAX_name + "/ao%d" % i
            self.AO_task.CreateAOVoltageChan(
                con, "", self.Vmin, self.Vmax, DAQmx_Val_Volts, None
            )

        # Setup DO channels
        '''for port_str in sorted(self.ports, key=split_conn_port):
            if not self.ports[port_str]['num_lines']:
                continue
            # Add each port to the task:
            con = '%s/%s' % (self.MAX_name, port_str)
            self.DO_task.CreateDOChan(con, "", DAQmx_Val_ChanForAllLines)'''
        self.DO_task.CreateDOChan(self.MAX_name+"/port0/line8:15",'', DAQmx_Val_ChanForAllLines)

        # Start tasks:
        if self.AO_task is not None:
            self.AO_task.StartTask()
        if self.DO_task is not None:
            self.DO_task.StartTask()
            pass

    def program_manual(self, front_panel_values):
        written = int32()
        if self.AO_task is not None:
            AO_data = np.zeros(self.num_AO, dtype=np.float64)
            for i in range(self.num_AO):
                AO_data[i] = front_panel_values['ao%d' % i]
            self.AO_task.WriteAnalogF64(
                1, True, 1, DAQmx_Val_GroupByChannel, AO_data, written, None
            )
        if self.DO_task is not None:
            # Due to two bugs in DAQmx, we will always pack our data into a uint32 and
            # write using WriteDigitalU32. The first bug is some kind of use of
            # uninitialised memory when using WriteDigitalLines, discussed here:
            # https://bitbucket.org/labscript_suite
            #     /labscript_devices/pull-requests/56/#comment-83671312
            # The second is that using a smaller int dtype sometimes fails even though
            # it is the correct int size for the size of the port. Using a 32 bit int
            # always works, the additional bits are ignored. This is discussed here:
            # https://forums.ni.com/t5/Multifunction-DAQ
            #     /problem-with-correlated-DIO-on-USB-6341/td-p/3344066
            DO_data = np.zeros(len(self.ports), dtype=np.uint32)
            for conn, value in front_panel_values.items():
                if conn.startswith('port'):
                    port, line = split_conn_DO(conn)
                    DO_data[port] |= value << line
            self.DO_task.WriteDigitalU32(
                1, True, 10.0, DAQmx_Val_GroupByChannel, DO_data, written, None
            )
        # TODO: return coerced/quantised values
        return {}

    def get_output_tables(self, h5file, device_name):
        """Return the AO and DO tables rom the file, or None if they do not exist."""
        with h5py.File(h5file, 'r') as hdf5_file:
            group = hdf5_file['devices'][device_name]
            try:
                AO_table = group['AO'][:]
            except KeyError:
                AO_table = None
            try:
                DO_table = group['DO'][:]
            except KeyError:
                DO_table = None
        return AO_table, DO_table

    def set_mirror_clock_terminal_connected(self, connected):
        """Mirror the clock terminal on another terminal to allow daisy chaining of the
        clock line to other devices, if applicable"""
        if self.clock_mirror_terminal is None:
            return
        if connected:
            DAQmxConnectTerms(
                self.clock_terminal,
                self.clock_mirror_terminal,
                DAQmx_Val_DoNotInvertPolarity,
            )
        else:
            DAQmxDisconnectTerms(self.clock_terminal, self.clock_mirror_terminal)

    def program_buffered_DO(self, DO_table):
        """Create the DO task and program in the DO table for a shot. Return a
        dictionary of the final values of each channel in use"""
        if DO_table is None:
            return {}
        self.DO_task = Task()
        written = int32()
        ports = DO_table.dtype.names

        final_values = {}
        for port_str in ports:
            # Add each port to the task:
            con = '%s/%s' % (self.MAX_name, port_str)
            self.DO_task.CreateDOChan(con, "", DAQmx_Val_ChanForAllLines)

            # Collect the final values of the lines on this port:
            port_final_value = DO_table[port_str][-1]
            for line in range(self.ports[port_str]["num_lines"]):
                # Extract each digital value from the packed bits:
                line_final_value = bool((1 << line) & port_final_value)
                final_values['%s/line%d' % (port_str, line)] = int(line_final_value)

        # Convert DO table to a regular array and ensure it is C continguous:
        DO_table = np.ascontiguousarray(
            structured_to_unstructured(DO_table, dtype=np.uint32)
        )

        # Check if DOs are all zero for the whole shot. If they are this triggers a
        # bug in NI-DAQmx that throws a cryptic error for buffered output. In this
        # case, run it as a non-buffered task.
        self.DO_all_zero = not np.any(DO_table)
        if self.DO_all_zero:
            DO_table = DO_table[0:1]

        if self.static_DO or self.DO_all_zero:
            # Static DO. Start the task and write data, no timing configuration.
            self.DO_task.StartTask()
            # Write data. See the comment in self.program_manual as to why we are using
            # uint32 instead of the native size of each port
            self.DO_task.WriteDigitalU32(
                1,  # npts
                False,  # autostart
                10.0,  # timeout
                DAQmx_Val_GroupByScanNumber,
                DO_table,
                written,
                None,
            )
        else:
            # We use all but the last sample (which is identical to the second last
            # sample) in order to ensure there is one more clock tick than there are
            # samples. This is required by some devices to determine that the task has
            # completed.
            npts = len(DO_table) - 1

            # Set up timing:
            self.DO_task.CfgSampClkTiming(
                self.clock_terminal,
                self.clock_limit,
                DAQmx_Val_Rising,
                DAQmx_Val_FiniteSamps,
                npts,
            )

            # Write data. See the comment in self.program_manual as to why we are using
            # uint32 instead of the native size of each port.
            self.DO_task.WriteDigitalU32(
                npts,
                False,  # autostart
                10.0,  # timeout
                DAQmx_Val_GroupByScanNumber,
                DO_table[:-1], # All but the last sample as mentioned above
                written,
                None,
            )
            # raise LabscriptError(DO_table[:-1])
            # Go!
            self.DO_task.StartTask()

        return final_values

    def program_buffered_AO(self, AO_table):
        if AO_table is None:
            return {}
        self.AO_task = Task()
        written = int32()
        channels = ', '.join(self.MAX_name + '/' + c for c in AO_table.dtype.names)
        self.AO_task.CreateAOVoltageChan(
            channels, "", self.Vmin, self.Vmax, DAQmx_Val_Volts, None
        )

        # Collect the final values of the analog outs:
        final_values = dict(zip(AO_table.dtype.names, AO_table[-1]))

        # Convert AO table to a regular array and ensure it is C continguous:
        AO_table = np.ascontiguousarray(
            structured_to_unstructured(AO_table, dtype=np.float64)
        )

        # Check if AOs are all zero for the whole shot. If they are this triggers a
        # bug in NI-DAQmx that throws a cryptic error for buffered output. In this
        # case, run it as a non-buffered task.
        self.AO_all_zero = not np.any(AO_table)
        if self.AO_all_zero:
            AO_table = AO_table[0:1]

        if self.static_AO or self.AO_all_zero:
            # Static AO. Start the task and write data, no timing configuration.
            self.AO_task.StartTask()
            self.AO_task.WriteAnalogF64(
                1, True, 10.0, DAQmx_Val_GroupByChannel, AO_table, written, None
            )
        else:
            # We use all but the last sample (which is identical to the second last
            # sample) in order to ensure there is one more clock tick than there are
            # samples. This is required by some devices to determine that the task has
            # completed.
            npts = len(AO_table) - 1

            # Set up timing:
            self.AO_task.CfgSampClkTiming(
                self.clock_terminal,
                self.clock_limit,
                DAQmx_Val_Rising,
                DAQmx_Val_FiniteSamps,
                npts,
            )

            # Write data:
            self.AO_task.WriteAnalogF64(
                npts,
                False,  # autostart
                10.0,  # timeout
                DAQmx_Val_GroupByScanNumber,
                AO_table[:-1],  # All but the last sample as mentioned above
                written,
                None,
            )

            # Go!
            self.AO_task.StartTask()

        return final_values

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        # Store the initial values in case we have to abort and restore them:
        self.initial_values = initial_values

        # Stop the manual mode output tasks, if any:
        self.stop_tasks()

        # Get the data to be programmed into the output tasks:
        AO_table, DO_table = self.get_output_tables(h5file, device_name)

        # Mirror the clock terminal, if applicable:
        self.set_mirror_clock_terminal_connected(True)

        # Program the output tasks and retrieve the final values of each output:
        DO_final_values = self.program_buffered_DO(DO_table)
        AO_final_values = self.program_buffered_AO(AO_table)

        final_values = {}
        final_values.update(DO_final_values)
        final_values.update(AO_final_values)

        # If we are the wait timeout device, then the final value of the timeout line
        # should be its rearm value:
        if self.wait_timeout_device == self.device_name:
            final_values[self.wait_timeout_connection] = self.wait_timeout_rearm_value

        return final_values

    def transition_to_manual(self, abort=False):
        # Stop output tasks and call program_manual. Only call StopTask if not aborting.
        # Otherwise results in an error if output was incomplete. If aborting, call
        # ClearTask only.
        npts = uInt64()
        samples = uInt64()
        tasks = []
        if self.AO_task is not None:
            tasks.append([self.AO_task, self.static_AO or self.AO_all_zero, 'AO'])
            self.AO_task = None
        if self.DO_task is not None:
            tasks.append([self.DO_task, self.static_DO or self.DO_all_zero, 'DO'])
            self.DO_task = None

        for task, static, name in tasks:
            if not abort:
                if not static:
                    try:
                        # Wait for task completion with a 1 second timeout:
                        task.WaitUntilTaskDone(-1)
                    finally:
                        # Log where we were up to in sample generation, regardless of
                        # whether the above succeeded:
                        task.GetWriteCurrWritePos(npts)
                        task.GetWriteTotalSampPerChanGenerated(samples)
                        # Detect -1 even though they're supposed to be unsigned ints, -1
                        # seems to indicate the task was not started:
                        current = samples.value if samples.value != 2 ** 64 - 1 else -1
                        total = npts.value if npts.value != 2 ** 64 - 1 else -1
                        msg = 'Stopping %s at sample %d of %d'
                        self.logger.info(msg, name, current, total)
                task.StopTask()
            task.ClearTask()

        # Remove the mirroring of the clock terminal, if applicable:
        self.set_mirror_clock_terminal_connected(False)

        # Set up manual mode tasks again:
        self.start_manual_mode_tasks()
        if abort:
            # Reprogram the initial states:
            self.program_manual(self.initial_values)

        return True

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        return self.transition_to_manual(True)

class NI_DAQmxAcquisitionWorker(Worker):
    MAX_READ_INTERVAL = 0.2
    MAX_READ_PTS = 10000

    def init(self):
        # Prevent interference between the read callback and the shutdown code:
        self.tasklock = threading.RLock()

        # Assigned on a per-task basis and cleared afterward:
        self.read_array = None
        self.task = None

        # Assigned on a per-shot basis and cleared afterward:
        self.buffered_mode = False
        self.h5_file = None
        self.acquired_data = None
        self.buffered_rate = None
        self.buffered_chans = None

        # Hard coded for now. Perhaps we will add functionality to enable
        # and disable inputs in manual mode, and adjust the rate:
        self.manual_mode_chans = self.AI_chans
        self.manual_mode_rate = 1000

        # An event for knowing when the wait durations are known, so that we may use
        # them to chunk up acquisition data:
        self.wait_durations_analysed = Event('wait_durations_analysed')

        # Start task for manual mode
        self.start_task(self.manual_mode_chans, self.manual_mode_rate)

    def shutdown(self):
        if self.task is not None:
            self.stop_task()

    def read(self, task_handle, event_type, num_samples, callback_data=None):
        """Called as a callback by DAQmx while task is running. Also called by us to get
        remaining data just prior to stopping the task. Since the callback runs
        in a separate thread, we need to serialise access to instance variables"""
        samples_read = int32()
        with self.tasklock:
            if self.task is None or task_handle != self.task.taskHandle.value:
                # Task stopped already.
                return 0
            self.task.ReadAnalogF64(
                num_samples,
                -1,
                DAQmx_Val_GroupByScanNumber,
                self.read_array,
                self.read_array.size,
                samples_read,
                None,
            )
            # Select only the data read, and downconvert to 32 bit:
            data = self.read_array[: int(samples_read.value), :].astype(np.float32)
            if self.buffered_mode:
                # Append to the list of acquired data:
                self.acquired_data.append(data)
            else:
                # TODO: Send it to the broker thingy.
                pass
        return 0

    def start_task(self, chans, rate):
        """Set up a task that acquires data with a callback every MAX_READ_PTS points or
        MAX_READ_INTERVAL seconds, whichever is faster. NI DAQmx calls callbacks in a
        separate thread, so this method returns, but data acquisition continues until
        stop_task() is called. Data is appended to self.acquired_data if
        self.buffered_mode=True, or (TODO) sent to the [whatever the AI server broker is
        called] if self.buffered_mode=False."""

        if self.task is not None:
            raise RuntimeError('Task already running')

        if chans is None:
            return

        # Get data MAX_READ_PTS points at a time or once every MAX_READ_INTERVAL
        # seconds, whichever is faster:
        num_samples = min(self.MAX_READ_PTS, int(rate * self.MAX_READ_INTERVAL))

        self.read_array = np.zeros((num_samples, len(chans)), dtype=np.float64)
        self.task = Task()

        if self.AI_term == 'RSE':
            term = DAQmx_Val_RSE
        elif self.AI_term == 'NRSE':
            term = DAQmx_Val_NRSE
        elif self.AI_term == 'Diff':
            term = DAQmx_Val_Diff
        elif self.AI_term == 'PseudoDiff':
            term = DAQmx_Val_PseudoDiff

        for chan in chans:
            self.task.CreateAIVoltageChan(
                self.MAX_name + '/' + chan,
                "",
                term,
                self.AI_range[0],
                self.AI_range[1],
                DAQmx_Val_Volts,
                None,
            )

        self.task.CfgSampClkTiming(
            "", rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, num_samples
        )
        if self.buffered_mode:
            self.task.CfgDigEdgeStartTrig(self.clock_terminal, DAQmx_Val_Rising)

        # This must not be garbage collected until the task is:
        self.task.callback_ptr = DAQmxEveryNSamplesEventCallbackPtr(self.read)

        self.task.RegisterEveryNSamplesEvent(
            DAQmx_Val_Acquired_Into_Buffer, num_samples, 0, self.task.callback_ptr, 100
        )

        self.task.StartTask()

    def stop_task(self):
        with self.tasklock:
            if self.task is None:
                raise RuntimeError('Task not running')
            # Read remaining data:
            self.read(self.task, None, -1)
            # Stop the task:
            self.task.StopTask()
            self.task.ClearTask()
            self.task = None
            self.read_array = None

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.logger.debug('transition_to_buffered')

        # read channels, acquisition rate, etc from H5 file
        with h5py.File(h5file, 'r') as f:
            group = f['/devices/' + device_name]
            if 'AI' not in group:
                # No acquisition
                return {}
            AI_table = group['AI'][:]
            device_properties = properties.get(f, device_name, 'device_properties')

        chans = [_ensure_str(c) for c in AI_table['connection']]
        # Remove duplicates and sort:
        if chans:
            self.buffered_chans = sorted(set(chans), key=split_conn_AI)
        self.h5_file = h5file
        self.buffered_rate = device_properties['acquisition_rate']
        if device_properties['start_delay_ticks']:
            # delay is defined in sample clock ticks, calculate in sec and save for later
            self.AI_start_delay = self.AI_start_delay_ticks*self.buffered_rate
        self.acquired_data = []
        # Stop the manual mode task and start the buffered mode task:
        self.stop_task()
        self.buffered_mode = True
        self.start_task(self.buffered_chans, self.buffered_rate)
        return {}

    def transition_to_manual(self, abort=False):
        self.logger.debug('transition_to_manual')
        #  If we were doing buffered mode acquisition, stop the buffered mode task and
        # start the manual mode task. We might not have been doing buffered mode
        # acquisition if abort() was called when we are not in buffered mode, or if
        # there were no acuisitions this shot.
        if not self.buffered_mode:
            return True
        if self.buffered_chans is not None:
            self.stop_task()
        self.buffered_mode = False
        self.logger.info('transitioning to manual mode, task stopped')
        self.start_task(self.manual_mode_chans, self.manual_mode_rate)

        if abort:
            self.acquired_data = None
            self.buffered_chans = None
            self.h5_file = None
            self.buffered_rate = None
            return True

        with h5py.File(self.h5_file, 'a') as hdf5_file:
            data_group = hdf5_file['data']
            data_group.create_group(self.device_name)
            waits_in_use = len(hdf5_file['waits']) > 0

        if self.buffered_chans is not None and not self.acquired_data:
            msg = """No data was acquired. Perhaps the acquisition task was not
                triggered to start, is the device connected to a pseudoclock?"""
            raise RuntimeError(dedent(msg))
        # Concatenate our chunks of acquired data and recast them as a structured
        # array with channel names:
        if self.acquired_data:
            start_time = time.time()
            dtypes = [(chan, np.float32) for chan in self.buffered_chans]
            raw_data = np.concatenate(self.acquired_data).view(dtypes)
            raw_data = raw_data.reshape((len(raw_data),))
            self.acquired_data = None
            self.buffered_chans = None
            self.extract_measurements(raw_data, waits_in_use)
            self.h5_file = None
            self.buffered_rate = None
            msg = 'data written, time taken: %ss' % str(time.time() - start_time)
        else:
            msg = 'No acquisitions in this shot.'
        self.logger.info(msg)

        return True

    def extract_measurements(self, raw_data, waits_in_use):
        self.logger.debug('extract_measurements')
        if waits_in_use:
            # There were waits in this shot. We need to wait until the other process has
            # determined their durations before we proceed:
            self.wait_durations_analysed.wait(self.h5_file)

        with h5py.File(self.h5_file, 'a') as hdf5_file:
            if waits_in_use:
                # get the wait start times and durations
                waits = hdf5_file['/data/waits']
                wait_times = waits['time']
                wait_durations = waits['duration']
            try:
                acquisitions = hdf5_file['/devices/' + self.device_name + '/AI']
            except KeyError:
                # No acquisitions!
                return
            try:
                measurements = hdf5_file['/data/traces']
            except KeyError:
                # Group doesn't exist yet, create it:
                measurements = hdf5_file.create_group('/data/traces')

            t0 = self.AI_start_delay
            for connection, label, t_start, t_end, _, _, _ in acquisitions:
                connection = _ensure_str(connection)
                label = _ensure_str(label)
                if waits_in_use:
                    # add durations from all waits that start prior to t_start of
                    # acquisition
                    t_start += wait_durations[(wait_times < t_start)].sum()
                    # compare wait times to t_end to allow for waits during an
                    # acquisition
                    t_end += wait_durations[(wait_times < t_end)].sum()
                i_start = int(np.ceil(self.buffered_rate * (t_start - t0)))
                i_end = int(np.floor(self.buffered_rate * (t_end - t0)))
                # np.ceil does what we want above, but float errors can miss the
                # equality:
                if t0 + (i_start - 1) / self.buffered_rate - t_start > -2e-16:
                    i_start -= 1
                # We want np.floor(x) to yield the largest integer < x (not <=):
                if t_end - t0 - i_end / self.buffered_rate < 2e-16:
                    i_end -= 1
                # IBS: we sometimes find that t_end (with waits) gives a time
                # after the end of acquisition.  The following line
                # will produce return a shorter than expected array if i_end
                # is larger than the length of the array.
                values = raw_data[connection][i_start : i_end + 1]
                i_end = i_start + len(values) - 1 # re-measure i_end

                t_i = t0 + i_start / self.buffered_rate
                t_f = t0 + i_end / self.buffered_rate
                times = np.linspace(t_i, t_f, len(values), endpoint=True)
                dtypes = [('t', np.float64), ('values', np.float32)]
                data = np.empty(len(values), dtype=dtypes)
                data['t'] = times
                data['values'] = values
                measurements.create_dataset(label, data=data)

    def abort_buffered(self):
        return self.transition_to_manual(True)

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def program_manual(self, values):
        return {}

class NI_DAQmxCounterAcquisitionWorker(Worker):
    def init(self):
        global h5py; import labscript_utils.h5_lock, h5py
        global threading; import threading
        global zprocess; import zprocess
        global logging; import logging
        global time; import time
        
        self.test_flag = False
        self.counter_task_running = False
        self.counter_daqlock = threading.Condition()
        # Channel details
        self.counter_channels = []

        self.counter_h5_file = "" 
        self.counter_buffered_channels = []
        self.CPT_buffered_channels = []
        self.trig_buffered_channels = []
        self.flag = 0

        self.counter_buffered = False
        
        # An event for knowing when the wait durations are known, so that we may use them
        # to chunk up acquisition data:
        self.wait_durations_analysed = zprocess.Event('wait_durations_analysed')
        
    def shutdown(self):
        if self.counter_task_running:
            self.counter_stop_task()
                    
    def setup_task(self):
        self.logger.info('setup_task') #Emily changed from debug to info
        #DAQmx Configure Code            
        self.flag += 1
        with self.counter_daqlock:
            self.logger.info('setup_task got daqlock')
            # Check if buffered and create correct channel lists
            if self.counter_buffered:
                counter_chnl_list = list(self.counter_buffered_channels)
                CPT_chnl_list = list(self.CPT_buffered_channels)
                trig_chnl_list = list(self.trig_buffered_channels)
            else: #TODO make this work if not buffered?
                counter_chnl_list = self.counter_channels
#                counter_rate = self.counter_rate  #EE2
            # Stop doing anything if there are no counter channels
            if len(counter_chnl_list) < 1:
                return 
            self.counter_read = [None]*len(counter_chnl_list)
            self.counter_data = [None]*len(counter_chnl_list)
            self.counter_samples_per_channel = [None]*len(counter_chnl_list)
            # self.counter_task_running = True

            for i, chnl in enumerate(counter_chnl_list):
                # self.logger.info(["counter chnl list", i, counter_chnl_list[i]])
                
                if self.counter_task[i]:
                    self.logger.info("clearing tasks")
                    self.counter_task[i].ClearTask()
                    self.pulser[i].ClearTask()

                self.counter_read[i] = int32() 
                samps_per_chan = 0
                max_samps_per_chan = 0
                sample_freq = self.counter_sample_freqs[chnl][0] #TODO ensure counter_sample_freqs[chnl] list contains only one number, because we cannot change the sample freq for different measurements during the shot
                for j in range(len(self.start_time[chnl])):
                    samps_per_chan += int(np.floor((self.end_time[chnl][j]-self.start_time[chnl][j])*sample_freq))
                    if int(np.floor((self.end_time[chnl][j]-self.start_time[chnl][j])*sample_freq)) > max_samps_per_chan:
                        max_samps_per_chan = int(np.floor((self.end_time[chnl][j]-self.start_time[chnl][j])*sample_freq))
                    # self.logger.info(["samps per chan", j, samps_per_chan, ' max samps per chan ', max_samps_per_chan])
                self.counter_data[i] = np.zeros(samps_per_chan, dtype=np.float64) # ejd 11/18

                ## ejd old ## 
                # self.pulser[i] = Task()
                # self.pulser[i].CreateCOPulseChanFreq("/Dev1/ctr1", '', DAQmx_Val_Hz, DAQmx_Val_Low, 0, sample_freq, 0.5)
                # self.pulser[i].CfgImplicitTiming(DAQmx_Val_FiniteSamps, samps_per_chan)
                # if self.counter_buffered:
                #     self.pulser[i].CfgDigEdgeStartTrig('/'+trig_chnl_list[i], DAQmx_Val_Rising)
                ## ejd new ##
                self.pulser[i] = Task()
                self.pulser[i].CreateCOPulseChanFreq("/Dev1/ctr1", '', DAQmx_Val_Hz, DAQmx_Val_Low, 0, sample_freq, 0.5) # TODO ejd does "/Dev1/ctr1" need to be a variable? ejd
                self.pulser[i].CfgImplicitTiming(DAQmx_Val_FiniteSamps, max_samps_per_chan) # TODO ejd this could be buggy...maybe should just require that all acquisitions are same length?
                if self.counter_buffered:
                    self.pulser[i].CfgDigEdgeStartTrig('/'+trig_chnl_list[i], DAQmx_Val_Rising) # does /Dev1/PFI4 need to be a variable? ejd
                    self.pulser[i].SetStartTrigRetriggerable(1)


                self.counter_task[i] = Task()
                self.logger.info(["Task", i, self.counter_task[i]])
                self.counter_task[i].CreateCICountEdgesChan('/' + counter_chnl_list[i], '', DAQmx_Val_Rising, 0, DAQmx_Val_CountUp) 
                self.counter_task[i].CfgSampClkTiming('/'+CPT_chnl_list[i], sample_freq, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, samps_per_chan)
                self.counter_task_running = True

                self.counter_task[i].StartTask()
                self.pulser[i].StartTask()

                #raise LabscriptError(self.counter_data[i])
#            self.pulser[i].RegisterDoneEvent(0, DAQmxDoneEventCallbackPtr callbackFunction, void *callbackData);
                #self.counter_task[i].ReadCounterF64(DAQmx_Val_Auto, -1, self.counter_data[i], self.counter_data[i].size, self.counter_read[i], None)
                #self.logger.info(self.counter_data[i])                
                #self.logger.info(len(self.counter_data[i]))
            self.counter_daqlock.notify() #Notify DAQmx_read that we are ready to acquire.
            
        self.logger.info('finished setup_task') #Emily changed from debug to info
        
    def stop_task(self):
        self.logger.info('stop_task') #Emily debug to info
        with self.counter_daqlock:
            self.logger.info('stop_task got daqlock')
            if self.counter_task_running:
                for i in range(len(self.counter_buffered_channels)): 
                    self.logger.info(["len(self.counter_buffered_channels", i, len(self.counter_buffered_channels)])
                    self.counter_task[i].ReadCounterF64(DAQmx_Val_Auto, -1, self.counter_data[i], self.counter_data[i].size, self.counter_read[i], None)
                    # self.logger.info(['b counter data', i, self.counter_data[i]])                
                    #self.logger.info(len(self.counter_data[i]))
                    #raise LabscriptError("ASDS")
                    self.counter_task[i].StopTask()
                    self.counter_task[i].ClearTask()
                    self.pulser[i].StopTask()
                    self.pulser[i].ClearTask()
                    self.counter_task_running = False #ejd 11/18 added

            self.counter_daqlock.notify()
        self.logger.debug('finished stop_task')
        
    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        # TODO: Do this line better!
        self.logger.debug('transitioning to buffer')
        self.counter_device_name = device_name

        # call stop_task function
        self.stop_task()
        
#        self.counter_buffered_data_list = [] #CO EE2
        # Set up dictionaries ejd 11/18
        self.start_time = {}
        self.end_time = {}
        self.counter_sample_freqs = {}
        # Save h5file path (for storing data later!)
        self.counter_h5_file = h5file ##
        # read channels, acquisition rate, etc from H5 file
        h5_count_chnls = []
        h5_CPT_chnls = []
        h5_trig_chnls = []        
        with h5py.File(h5file,'r') as hdf5_file:
            group =  hdf5_file['/devices/'+device_name]
            if 'counter_channels' in group.attrs: 
                h5_count_chnls = group.attrs['counter_channels'].split(', ')
                # self.logger.info(['h5_count_chnls', h5_count_chnls, set(h5_count_chnls)])
                h5_CPT_chnls = group.attrs['cpt_channels'].split(', ') ##EE2
                h5_trig_chnls = group.attrs['trig_channels'].split(', ')##EE2
#                self.counter_buffered_rate = float(group.attrs['counter_acquisition_rate']) ## Emily edit: Not sure if I need this for the counter...
            else:
                self.logger.info("no counter channels")
            ## ejd commented out start 11/18
            # try:
            #     counter_acquisitions = hdf5_file['/devices/'+device_name+'/COUNTER_ACQUISITIONS']
            #     self.start_time = [float(counter_acquisitions[i][4]) for i in range(len(counter_acquisitions))]
            #     self.end_time= [float(counter_acquisitions[i][5]) for i in range(len(counter_acquisitions))]
            #     self.counter_sample_freqs = [float(counter_acquisitions[i][6]) for i in range(len(counter_acquisitions))]
            ## ejd 11/18 commented out end
            ## ejd 11/18 replacement start
            # try:
            counter_acquisitions = hdf5_file['/devices/'+device_name+'/COUNTER_ACQUISITIONS']
            self.numIterations = int(counter_acquisitions[0][-1])
            # self.logger.info(['self.numIterations ', self.numIterations])
            for j, chnl in enumerate([counter_acquisitions[i][0] for i in range(len(counter_acquisitions))]):
                chnl = chnl.decode("utf-8")
                # self.logger.info(['device name ', device_name, ' chanl ', chnl])
                if device_name+'/'+chnl not in self.start_time:
                    self.start_time[device_name+'/'+chnl] = [float(counter_acquisitions[j][4])]
                    self.end_time[device_name+'/'+chnl] = [float(counter_acquisitions[j][5])]
                    self.counter_sample_freqs[device_name+'/'+chnl] = [float(counter_acquisitions[j][6])]
                else:
                    self.start_time[device_name+'/'+chnl].append(float(counter_acquisitions[j][4]))# = [float(counter_acquisitions[i][4]) for i in range(len(counter_acquisitions))] # changed to dict ejd
                    self.end_time[device_name+'/'+chnl].append(float(counter_acquisitions[j][5]))#= [float(counter_acquisitions[i][5]) for i in range(len(counter_acquisitions))] #changed to dict ejd
                    self.counter_sample_freqs[device_name+'/'+chnl].append(float(counter_acquisitions[j][6])) #= [float(counter_acquisitions[i][6]) for i in range(len(counter_acquisitions))]  #changed to dict ejd
                # self.logger.info([self.start_time, self.end_time, self.counter_sample_freqs])
            for chnl in h5_count_chnls:
                if not all(element == self.counter_sample_freqs[chnl][0] for element in self.counter_sample_freqs[chnl]):
                    self.logger.debug('WARNING COUNTER SAMPLE FREQ NEEDS TO BE THE SAME FOR ALL ACQUISITIONS ON A SINGLE COUNTER DURING THE SHOT!') #TODO ejd labscript error, also possibly TODO ejd check what happens if all acquisitions are not the same length? raise LabscriptError()
            ## ejd 11/18 replacement end              
            # except:
            #     # No acquisitions!
            #     pass
            ## ejd 11/18 commented out start
            # h5_count_chnls = [self.MAX_name +'/'+ counter_acquisitions[i][0].decode("utf-8") for i in range(len(counter_acquisitions))]
            # self.logger.info([counter_acquisitions[i][0].decode("utf-8") for i in range(len(counter_acquisitions))])
            # h5_CPT_chnls = [self.MAX_name +'/'+counter_acquisitions[i][1].decode("utf-8") for i in range(len(counter_acquisitions))] ##EE2
            # h5_trig_chnls = [self.MAX_name +'/'+counter_acquisitions[i][2].decode("utf-8") for i in range(len(counter_acquisitions))]##EE2
            ## ejd 11/18 commented out end



        # combine static channels with h5 channels (use a set to avoid duplicates?)
        #raise LabscriptError(h5_count_chnls)
        self.counter_buffered_channels = h5_count_chnls ## set?
        # self.logger.info(['counter_buffered_channels', self.counter_buffered_channels, set(self.counter_buffered_channels)])
        self.CPT_buffered_channels = h5_CPT_chnls ## use set here?! Multiple counters could use same CPT counter. ##EE2
        self.trig_buffered_channels = h5_trig_chnls ## use set here?! Multiple counters could use same CPT counter. ##EE2
        self.counter_buffered = True 

        self.counter_task = [None]*len(h5_count_chnls) ##EE2
        self.pulser = [None]*len(h5_count_chnls) ##EE2
        self.counter_abort = [None]*len(h5_count_chnls) ##EE2
        self.trigger = [None]*len(h5_count_chnls) ##EE2

        # unclear what this is doing...EE2
#        self.counter_buffered_channels.update(self.counter_channels)
        
        self.logger.info('Calling setup task')
        #raise LabscriptError("ASDSADA")
        self.setup_task()   
        return {}
    
    def transition_to_manual(self,abort=False):    
        self.logger.debug('transition_to_static')
        # Stop acquisition (this should really be done on a digital edge, but that is for later! Maybe use a Counter)
        # Set the abort flag so that the acquisition thread knows to expect an exception in the case of an abort:
        #
        # TODO: This is probably bad because it shortly gets overwritten to False
        # However whether it has an effect depends on whether daqmx_read thread holds the daqlock 
        # when self.stop_task() is called
        self.counter_abort = abort
        self.test_flag = True
        self.stop_task()
        # Reset the abort flag so that unexpected exceptions are still raised:        
        self.counter_abort = False
        self.logger.info('transitioning to static, task stopped')
        # save the data acquired to the h5 file
        if not abort:
#            with h5py.File(self.counter_h5_file,'a') as hdf5_file: ## Emily: I don't think I need this
#                data_group = hdf5_file['data']
#                data_group.create_group(self.device_name)

            counter_dtypes = [(chan.split('/')[-1],np.float32) for chan in sorted(self.counter_buffered_channels)]
            start_time = time.time()
            if len(self.counter_buffered_channels) != 0:
                self.extract_measurements(self.device_name)
                self.logger.info('data written, time taken: %ss' % str(time.time()-start_time))
            
#            self.counter_buffered_data = None #CO EE2
#            self.counter_buffered_data_list = []
            
            # Send data to callback functions as requested (in one big chunk!)
            #self.result_queue.put([self.t0,self.rate,self.ai_read,len(self.channels),self.ai_data])
        
        # return to previous acquisition mode
        self.counter_buffered = False #[False]*len(self.counter_buffered_channels) ##EE2
        self.setup_task()
        
        return True
        
    def extract_measurements(self, device_name):
        self.logger.info('extract_measurements') #Emily debug==>info
        self.logger.info(self.counter_h5_file)
        with h5py.File(self.counter_h5_file,'a') as hdf5_file:
            counter_waits_in_use = len(hdf5_file['waits']) > 0
        if counter_waits_in_use:
            # There were waits in this shot. We need to wait until the other process has
            # determined their durations before we proceed:
            self.wait_durations_analysed.wait(self.counter_h5_file)
        with h5py.File(self.counter_h5_file,'a') as hdf5_file:
            try:
                counter_acquisitions = hdf5_file['/devices/'+device_name+'/COUNTER_ACQUISITIONS']
            except:
                # No acquisitions!
                return
            try:
                counter_measurements = hdf5_file['/data/counter']
            except:
                # Group doesn't exist yet, create it: TODO NEED TO EDIT IF >1 Counter Channel
                counter_measurements = hdf5_file.create_group('/data/counter')
            counter_measurements.create_dataset('allCounterData', data=self.counter_data)
            counter_index = 0
            indStart = 0
            indEnd = 0
            # for counter_connection,counter_label,counter_start_time,counter_end_time,sample_freq,counter_wait_label,counter_scale_factor,counter_units in counter_acquisitions: # original but wrong ejd
            for counter_connection,counter_CPT_connection,counter_trigger,counter_label,counter_start_time,counter_end_time, counter_sample_freq, counter_wait_label, numIter in counter_acquisitions: #TODO this won't work if more than one counter channel ejd
                num_samps = int(np.floor((counter_end_time-counter_start_time)*counter_sample_freq))
                # self.logger.info(['num sampps', num_samps, ' counter end time ', counter_end_time, ' counter start time ', counter_start_time, ' counter sample freq ', counter_sample_freq])

                counter_label = _ensure_str(counter_label)
                counter_label = counter_label + '_' + str(counter_index)
                indEnd += num_samps
                # self.logger.info(self.counter_data[0])
                # self.logger.info(self.counter_data[0][indStart:indEnd])
                # counter_measurements.create_dataset(counter_label, data=self.counter_data) #ejd original
                counter_measurements.create_dataset(counter_label, data=self.counter_data[0][indStart:indEnd])
                indStart = indEnd
                counter_index += 1
    def abort_buffered(self):
        #TODO: test this
        return self.transition_to_manual(True)
        
    def abort_transition_to_buffered(self):
        #TODO: test this
        return self.transition_to_manual(True)   
    
    def program_manual(self,values):
        return {}
 

class NI_DAQmxWaitMonitorWorker(Worker):
    def init(self):

        self.all_waits_finished = Event('all_waits_finished', type='post')
        self.wait_durations_analysed = Event('wait_durations_analysed', type='post')
        self.wait_completed = Event('wait_completed', type='post')

        # Set on a per-shot basis and cleared afterward:
        self.h5_file = None
        self.CI_task = None
        self.DO_task = None
        self.wait_table = None
        self.semiperiods = None
        self.wait_monitor_thread = None

        # Saved error in case one occurs in the thread, we can raise it later in
        # transition_to_manual:
        self.wait_monitor_thread_exception = None
        # To trigger early shutdown of the wait monitor thread:
        self.shutting_down = False

        # Does this device have the "incomplete sample detection" feature? This
        # determines whether the first sample on our semiperiod counter input task will
        # be automatically discarded before we see it, or whether we will have to
        # discard it ourselves
        self.incomplete_sample_detection = incomplete_sample_detection(self.MAX_name)

        # Data for timeout triggers:
        if self.timeout_trigger_type == 'rising':
            trigger_value = 1
            rearm_value = 0
        elif self.timeout_trigger_type == 'falling':
            trigger_value = 0
            rearm_value = 1
        else:
            msg = 'timeout_trigger_type  must be "rising" or "falling", not "{}".'
            raise ValueError(msg.format(self.timeout_trigger_type))
        self.timeout_trigger = np.array([trigger_value], dtype=np.uint8)
        self.timeout_rearm = np.array([rearm_value], dtype=np.uint8)

    def shutdown(self):
        self.stop_tasks(True)

    def read_edges(self, npts, timeout=None):
        """Wait up to the given timeout in seconds for an edge on the wait monitor and
        and return the duration since the previous edge. Return None upon timeout."""
        samples_read = int32()
        # If no timeout, call read repeatedly with a 0.2 second timeout to ensure we
        # don't block indefinitely and can still abort.
        if timeout is None:
            read_timeout = 0.2
        else:
            read_timeout = timeout
        read_array = np.zeros(npts)
        while True:
            if self.shutting_down:
                raise RuntimeError('Stopped before expected number of samples acquired')
            try:
                self.CI_task.ReadCounterF64(
                    npts, read_timeout, read_array, npts, samples_read, None
                )
            except SamplesNotYetAvailableError:
                if timeout is None:
                    continue
                return None
            return read_array

    def wait_monitor(self):
        try:
            # Read edge times from the counter input task, indiciating the times of the
            # pulses that occur at the start of the experiment and after every wait. If a
            # timeout occurs, pulse the timeout output to force a resume of the master
            # pseudoclock. Save the resulting
            self.logger.debug('Wait monitor thread starting')
            with self.kill_lock:
                self.logger.debug('Waiting for start of experiment')
                # Wait for the pulse indicating the start of the experiment:
                if self.incomplete_sample_detection:
                    semiperiods = self.read_edges(1, timeout=None)
                else:
                    semiperiods = self.read_edges(2, timeout=None)
                self.logger.debug('Experiment started, got edges:' + str(semiperiods))
                # May have been one or two edges, depending on whether the device has
                # incomplete sample detection. We are only interested in the second one
                # anyway, it tells us how long the initial pulse was. Store the pulse width
                # for later, we will use it for making timeout pulses if necessary. Note
                # that the variable current_time is labscript time, so it will be reset
                # after each wait to the time of that wait plus pulse_width.
                current_time = pulse_width = semiperiods[-1]
                self.semiperiods.append(semiperiods[-1])
                # Alright, we're now a short way into the experiment.
                for wait in self.wait_table:
                    # How long until when the next wait should timeout?
                    timeout = wait['time'] + wait['timeout'] - current_time
                    timeout = max(timeout, 0)  # ensure non-negative
                    # Wait that long for the next pulse:
                    self.logger.debug('Waiting for pulse indicating end of wait')
                    semiperiods = self.read_edges(2, timeout)
                    # Did the wait finish of its own accord, or time out?
                    if semiperiods is None:
                        # It timed out. If there is a timeout device, send a trigger to
                        # resume the clock!
                        if self.DO_task is not None:
                            msg = """Wait timed out; retriggering clock with {:.3e} s
                                pulse ({} edge)"""
                            msg = msg.format(pulse_width, self.timeout_trigger_type)
                            self.logger.debug(dedent(msg))
                            self.send_resume_trigger(pulse_width)
                        else:
                            msg = """Specified wait timeout exceeded, but there is no
                                timeout device with which to resume the experiment.
                                Continuing to wait."""
                            self.logger.warning(dedent(msg))
                        # Keep waiting for the clock to resume:
                        self.logger.debug('Waiting for pulse indicating end of wait')
                        semiperiods = self.read_edges(2, timeout=None)
                    # Alright, now we're at the end of the wait.
                    self.semiperiods.extend(semiperiods)
                    self.logger.debug('Wait completed')
                    current_time = wait['time'] + semiperiods[-1]
                    # Inform any interested parties that a wait has completed:
                    postdata = _ensure_str(wait['label'])
                    self.wait_completed.post(self.h5_file, data=postdata)
                # Inform any interested parties that waits have all finished:
                self.logger.debug('All waits finished')
                self.all_waits_finished.post(self.h5_file)
        except Exception:
            self.logger.exception('Exception in wait monitor thread:')
            # Save the exception so it can be raised in transition_to_manual
            self.wait_monitor_thread_exception = sys.exc_info()

    def send_resume_trigger(self, pulse_width):
        written = int32()
        # Trigger:
        self.DO_task.WriteDigitalLines(
            1, True, 1, DAQmx_Val_GroupByChannel, self.timeout_trigger, written, None
        )
        # Wait however long we observed the first pulse of the experiment to be. In
        # practice this is likely to be negligible compared to the other software delays
        # here, but in case it is larger we'd better wait:
        time.sleep(pulse_width)
        # Rearm trigger:
        self.DO_task.WriteDigitalLines(
            1, True, 1, DAQmx_Val_GroupByChannel, self.timeout_rearm, written, None
        )

    def stop_tasks(self, abort):
        self.logger.debug('stop_tasks')
        if self.wait_monitor_thread is not None:
            if abort:
                # This will cause the wait_monitor thread to raise an exception within a
                # short time, allowing us to join it before it would otherwise be done.
                self.shutting_down = True
            self.wait_monitor_thread.join()
            self.wait_monitor_thread = None
            self.shutting_down = False
            if not abort and self.wait_monitor_thread_exception is not None:
                # Raise any unexpected errors from the wait monitor thread:
                _reraise(*self.wait_monitor_thread_exception)
            self.wait_monitor_thread_exception = None
            if not abort:
                # Don't want errors about incomplete task to be raised if we are aborting:
                self.CI_task.StopTask()
            if self.DO_task is not None:
                self.DO_task.StopTask()
        if self.CI_task is not None:
            self.CI_task.ClearTask()
            self.CI_task = None
        if self.DO_task is not None:
            self.DO_task.ClearTask()
            self.DO_task = None
        self.logger.debug('finished stop_tasks')

    def start_tasks(self):

        # The counter acquisition task:
        self.CI_task = Task()
        CI_chan = self.MAX_name + '/' + self.wait_acq_connection
        # What is the longest time in between waits, plus the timeout of the
        # second wait?
        interwait_times = np.diff([0] + list(self.wait_table['time']))
        max_measure_time = max(interwait_times + self.wait_table['timeout'])
        # Allow for software delays in timeouts.
        max_measure_time += 1.0
        min_measure_time = self.min_semiperiod_measurement
        self.logger.debug(
            "CI measurement range is: min: %f max: %f",
            min_measure_time,
            max_measure_time,
        )
        self.CI_task.CreateCISemiPeriodChan(
            CI_chan, '', min_measure_time, max_measure_time, DAQmx_Val_Seconds, ""
        )
        num_edges = 2 * (len(self.wait_table) + 1)
        self.CI_task.CfgImplicitTiming(DAQmx_Val_ContSamps, num_edges)
        self.CI_task.StartTask()

        # The timeout task:
        if self.wait_timeout_MAX_name is not None:
            self.DO_task = Task()
            DO_chan = self.wait_timeout_MAX_name + '/' + self.wait_timeout_connection
            self.DO_task.CreateDOChan(DO_chan, "", DAQmx_Val_ChanForAllLines)
            # Ensure timeout trigger is armed:
            written = int32()
            # Writing autostarts the task:
            self.DO_task.WriteDigitalLines(
                1, True, 1, DAQmx_Val_GroupByChannel, self.timeout_rearm, written, None
            )

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        self.logger.debug('transition_to_buffered')
        self.h5_file = h5file
        with h5py.File(h5file, 'r') as hdf5_file:
            dataset = hdf5_file['waits']
            if len(dataset) == 0:
                # There are no waits. Do nothing.
                self.logger.debug('There are no waits, not transitioning to buffered')
                self.wait_table = None
                return {}
            self.wait_table = dataset[:]

        self.start_tasks()

        # An array to store the results of counter acquisition:
        self.semiperiods = []
        self.wait_monitor_thread = threading.Thread(target=self.wait_monitor)
        # Not a daemon thread, as it implements wait timeouts - we need it to stay alive
        # if other things die.
        self.wait_monitor_thread.start()
        self.logger.debug('finished transition to buffered')

        return {}

    def transition_to_manual(self, abort=False):
        self.logger.debug('transition_to_manual')
        self.stop_tasks(abort)
        if not abort and self.wait_table is not None:
            # Let's work out how long the waits were. The absolute times of each edge on
            # the wait monitor were:
            edge_times = np.cumsum(self.semiperiods)
            # Now there was also a rising edge at t=0 that we didn't measure:
            edge_times = np.insert(edge_times, 0, 0)
            # Ok, and the even-indexed ones of these were rising edges.
            rising_edge_times = edge_times[::2]
            # Now what were the times between rising edges?
            periods = np.diff(rising_edge_times)
            # How does this compare to how long we expected there to be between the
            # start of the experiment and the first wait, and then between each pair of
            # waits? The difference will give us the waits' durations.
            resume_times = self.wait_table['time']
            # Again, include the start of the experiment, t=0:
            resume_times = np.insert(resume_times, 0, 0)
            run_periods = np.diff(resume_times)
            wait_durations = periods - run_periods
            waits_timed_out = wait_durations > self.wait_table['timeout']

            # Work out how long the waits were, save them, post an event saying so:
            dtypes = [
                ('label', 'a256'),
                ('time', float),
                ('timeout', float),
                ('duration', float),
                ('timed_out', bool),
            ]
            data = np.empty(len(self.wait_table), dtype=dtypes)
            data['label'] = self.wait_table['label']
            data['time'] = self.wait_table['time']
            data['timeout'] = self.wait_table['timeout']
            data['duration'] = wait_durations
            data['timed_out'] = waits_timed_out
            with h5py.File(self.h5_file, 'a') as hdf5_file:
                hdf5_file.create_dataset('/data/waits', data=data)
            self.wait_durations_analysed.post(self.h5_file)

        self.h5_file = None
        self.semiperiods = None
        return True

    def abort_buffered(self):
        return self.transition_to_manual(True)

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def program_manual(self, values):
        return {}