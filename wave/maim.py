import level
import numpy as np
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

def is_open(pin):
    return GPIO.input(pin)

door = 15
GPIO.setup(door, GPIO.IN)

t = []
voltage = []

mcp = level.MCP3021(5.2, 0)
print('start')
while is_open(door) == 1:
    time.sleep(0.01)
print('open')
start = time.time()
while time.time() - start <= 15:
    voltage.append(mcp.get_voltage())
    t.append(time.time() - start)
    time.sleep(0.01)

data= np.column_stack((t, voltage))

np.savetxt('w80.csv', data, delimiter=',', fmt='%.4f', header='с,В', comments='', encoding="utf8")