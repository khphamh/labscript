import serial as serial
import time as time
ZCU4ser = serial.Serial("com5", baudrate=115200, timeout=1)


ZCU4ser.write(b"cd jupyter_notebooks\r\n")

ZCU4ser.write(b"cd qick\r\n")

ZCU4ser.write(b"cd qick_demos\r\n")

ZCU4ser.write(b"sudo python3\r\n")
time.sleep(2)

ZCU4ser.write(b"xilinx\r\n")
time.sleep(2)

ZCU4ser.write(b"exec(open('initialize.py').read())\r\n")
time.sleep(15)
print(ZCU4ser.read(ZCU4ser.inWaiting()).decode())

pulse_list = [[5,'buffer',0, 200, 32766,100,0,'oneshot','product','[]'], [6,'buffer',0, 200, 32766,200,90,'oneshot','product','[]']]
pulse_list_string = "pulse_list = " + str(pulse_list) + "\r\n"
ZCU4ser.write(pulse_list_string.encode())
sequence_list = [(4,0,10)]
sequence_list_string = "sequence_list = " + str(sequence_list) + "\r\n"
ZCU4ser.write(sequence_list_string.encode())
loop_number_string = "number_of_loops = " + str(2) + "\r\n"
ZCU4ser.write(loop_number_string.encode())
delay_time_string = "delay_time = " + str(0) + "\r\n"
ZCU4ser.write(delay_time_string.encode())
start_src_string = "start = " + '\''+ 'external' + '\''+ "\r\n"
ZCU4ser.write(start_src_string.encode())
ZCU4ser.write(b"exec(open('send_pulse.py').read())\r\n")
ZCU4ser.write(b"exec(open('send_pulse.py').read())\r\n")

print(ZCU4ser.read(ZCU4ser.inWaiting()).decode())
