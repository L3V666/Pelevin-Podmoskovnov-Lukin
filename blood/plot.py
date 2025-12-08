from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/rest.csv'))
data_clean = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/data_clean_med.csv')) 

plt.plot(data_clean[:, 0], data_clean[:, 1], label='data_clean')

local_min = []

s = 1000
for i in range(s, len(data_clean[:, 1]) - s):
    if np.min(data_clean[i - s:i, 1]) > data_clean[i, 1] < np.min(data_clean[i+1:i + s, 1]):
        local_min.append([data_clean[i, 0], data_clean[i, 1]])


#print(local_min)
local_min = np.array(local_min)


deg = 4
coeffs = np.polyfit(local_min[:, 0], local_min[:, 1], deg)

p = np.poly1d(coeffs)
x = data_clean[:, 0]
y_fit = p(x)

#print(p)

#plt.plot(x, y_fit)

#plt.plot(data_clean[:, 0], data_clean[:, 1] - y_fit)
m = data_clean[54446:90634, 1] - y_fit[54446:90634]
n = data_clean[54446:90634, 0]
np.savetxt('30-50.csv', np.column_stack((n, m)), delimiter=',', fmt='%.4f', header='с,В', comments='', encoding="utf8")

plt.legend()

plt.show()