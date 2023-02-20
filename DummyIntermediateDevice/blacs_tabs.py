from blacs.device_base_class import DeviceTab, define_state, MODE_BUFFERED
class DummyIntermediateDeviceTab(DeviceTab):
    def initialise_GUI(self):
        self.create_worker("main_worker","labscript_devices.DummyIntermediateDevice.blacs_workers.DummyIntermediateDeviceWorker",{})
        self.primary_worker = "main_worker"