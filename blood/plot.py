from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data_clean = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/pridurok_clean.csv')) 

data_clean[:, 1] *= 80.65 

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

plt.plot(x, y_fit)

#plt.plot(data_clean[:, 0], data_clean[:, 1] - y_fit)
m = data_clean[54490:90782, 1] - y_fit[54490:90782]
n = data_clean[54490:90782, 0]
np.savetxt('30-50pridurok.csv', np.column_stack((n, m)), delimiter=',', fmt='%.4f', header='с,В', comments='', encoding="utf8")

plt.xlim(0, 60)

plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

plt.minorticks_on()

plt.ylabel('$P$, мм рт. ст.')
plt.xlabel('$t$, с')

plt.show()