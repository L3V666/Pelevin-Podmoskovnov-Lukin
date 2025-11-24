from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = np.array(pd.read_csv('w40.csv'))

plt.scatter(data[:, 0], data[:, 1])

plt.show()