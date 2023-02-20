import labscript_devices

labscript_device_name = 'DummyIntermediateDevice'
blacs_tab = 'labscript_devices.DummyIntermediateDevice.blacs_tabs.DummyIntermediateDeviceTab'

labscript_devices.register_classes(
    labscript_device_name=labscript_device_name,
    BLACS_tab=blacs_tab
)