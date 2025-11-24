from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

data = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/wave/w80.csv')[1:])
data[:, 1] /= 0.026

def linear_func(x, a, b):
    return a * x + b

def lsm(x, y): 
    params, covariance = curve_fit(linear_func, x, y)
    xs = np.linspace(0, 2 * max(x), 1000)
    ys = linear_func(xs, params[0], params[1])
    a, b = params
    da, db = np.sqrt(np.diag(covariance))
    return xs, ys, a, b, da, db


plt.scatter(data[:, 0], data[:, 1], s=1, color='red')

x, y, a, b, da, db = lsm(data[:90, 0], data[:90, 1])

plt.plot(x, y, linewidth=2.5, c='dodgerblue')

x, y, a, b, da, db = lsm(data[150:160, 0], data[150:160, 1])

plt.plot(x, y, linewidth=2.5, c='dodgerblue')

x0 = np.array([1.32, 1.32])
y0 = np.array([0, 90])

plt.plot(x0, y0, linestyle='--', c='green')

plt.xlim(0, 15)
plt.ylim(0, 90)

plt.xlabel('t, с')
plt.ylabel('h, мм')


plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

plt.minorticks_on()

plt.legend()

plt.savefig('w80.png')

plt.show()