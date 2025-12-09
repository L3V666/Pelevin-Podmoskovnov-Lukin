from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/30-50clean.csv')[1:])

data[:, 1] *= 80.65

plt.plot(data[:, 0], data[:, 1])

plt.ylabel('$\Delta P$, мм рт. ст.')
plt.xlabel('$t$, с')

plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

plt.minorticks_on()

plt.show()