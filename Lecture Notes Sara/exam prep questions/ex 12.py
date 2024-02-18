import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

xdata= np.linspace(0, 8*np.pi, 25)
ydata = (50*np.sin(xdata))/(xdata+1)

splfun = CubicSpline(xdata,ydata)

x=np.linspace(0, 8*np.pi, 25)
y=splfun(x)
plt.plot(x,y)
plt.plot(xdata,ydata,'o')
plt.legend()
plt.show()
