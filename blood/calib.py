from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def linear_func(x, a, b):
    return a * x + b

def lsm(x, y): 
    params, covariance = curve_fit(linear_func, x, y)
    xs = np.linspace(0, 2 * max(x), 1000)
    ys = linear_func(xs, params[0], params[1])
    a, b = params
    da, db = np.sqrt(np.diag(covariance))
    return xs, ys, a, b, da, db

data1 = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/40mmHg.csv')[1:])
data2 = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/80mmHg.csv')[1:])
data3 = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/120mmHg.csv')[1:])
data4 = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/160mmHg.csv')[1:])

data = np.array([np.mean(data1[:, 1]), np.mean(data2[:, 1]), np.mean(data3[:, 1]), np.mean(data4[:, 1])])

plt.scatter(data, [40, 80, 120, 160], marker='+', color='red', s=100)

x, y, a, b, da, db = lsm(data, [40, 80, 120, 160])
plt.plot(x, y, label=fr'$P={a:.4f}\cdot U{b:.4f}$')

plt.legend()

plt.ylabel('$P$, мм рт. ст.')
plt.xlabel('$U$, В')

plt.xlim(0, 2.5)
plt.ylim(0, 180)

plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

plt.minorticks_on()

plt.show()
