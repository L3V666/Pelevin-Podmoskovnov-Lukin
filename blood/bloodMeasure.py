import bloodFunctions as mcp
import time
import numpy as np

adc = mcp.MCP3021(5.18)
voltage_values=[]
time_values=[]
duration = 60.0
try:
    time_start=time.time()
    while time.time()-time_start<=duration:
        time_current = time.time()
        voltage_values.append(adc.get_voltage())
        time_values.append(time_current-time_start)
    data= np.column_stack((time_values, voltage_values))
    np.savetxt('rest.csv', data, delimiter=',', fmt='%.4f', header='с,В', comments='', encoding="utf8")
finally:
    adc.deinit()