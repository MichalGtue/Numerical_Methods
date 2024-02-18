import scipy.interpolate
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
x = np.array([2,4,4.5,5.8])
y = np.array(([4,4,9,-12]))

sol = np.polyfit(x,y,4)
x_new = np.linspace(1.5,6,300)

y_plot = np.polyval(sol,x_new)

plt.scatter(x,y, label=' data')

plt.plot(x_new, y_plot, label='p')

print(sol)

plt.show()