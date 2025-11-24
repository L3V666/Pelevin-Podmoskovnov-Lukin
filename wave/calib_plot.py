from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

data = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/wave/calib.csv')[1:])

def linear_func(x, a, b):
    return a * x + b

def lsm(x, y): 
    params, covariance = curve_fit(linear_func, x, y)
    xs = np.linspace(0, 2 * max(x), 1000)
    ys = linear_func(xs, params[0], params[1])
    a, b = params
    da, db = np.sqrt(np.diag(covariance))
    return xs, ys, a, b, da, db


plt.scatter(data[:, 0], data[:, 1], marker='+', s=200, color='red')

x, y, a, b, da, db = lsm(data[:, 0], data[:, 1])

plt.xlim(0, 120)
plt.ylim(0, 4)

plt.xlabel('h, мм')
plt.ylabel('U, В')


plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

plt.minorticks_on()

plt.plot(x, y, label=fr'$U={a:.4f}\cdot h {b:.4f}$')

plt.legend()

plt.savefig('calib.png')

plt.show()