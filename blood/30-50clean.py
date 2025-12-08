from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = np.array(pd.read_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/30-50clean.csv')[1:])

plt.plot(data[:, 0], data[:, 1])

plt.show()