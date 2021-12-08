import matplotlib.pyplot as plt
import numpy as np

# Define X and Y variable data
x = np.array([10, 20, 50, 100, 200, 400])
y = np.array([0.15786, 0.16303, 0.15714, 0.21770, 0.21957, 0.21923])

plt.plot(x, y)
plt.xlabel("Batch Size")  # add X-axis label
plt.ylabel("RMSLE")  # add Y-axis label
plt.show()