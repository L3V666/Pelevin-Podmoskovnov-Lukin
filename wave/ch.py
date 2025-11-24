from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def linear_func(x, a, b):
    return a * x + b

def lsm(x, y): 
    params, covariance = curve_fit(linear_func, x, y)
    xs = np.linspace(0, max(x), 1000)
    ys = linear_func(xs, params[0], params[1])
    a, b = params
    da, db = np.sqrt(np.diag(covariance))
    return xs, ys, a, b, da, db

x = np.array([40, 80, 105])
y = np.array([0.4, 1.04, 1.37])

plt.scatter(x, y, marker='+', s=200, c='red')

x, y, a, b, da, db = lsm(x, y)

plt.plot(x, y, linewidth=2.5, c='dodgerblue')

plt.xlim(0, 110)
plt.ylim(0, 1.5)

plt.ylabel('$c^2, м^2/c^2$')
plt.xlabel('$h, мм$')

plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

plt.minorticks_on()

plt.savefig('ch.png')

plt.show()