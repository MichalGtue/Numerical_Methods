import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
x = np.linspace(0,8*np.pi, 25)

x_high =  np.linspace(0,8*np.pi, 50)

y = 50*np.sin(x)/(x+1)
y_high = 50*np.sin(x_high)/(x_high+1)
ifun = make_interp_spline(x_high,y_high)
plt.plot(x,y, label='exact')
plt.plot(x_high,y_high, label='spline')
plt.legend()

plt.show()