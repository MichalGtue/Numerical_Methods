import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

## Interpolation

x = np.arange(0,6)
ydata = x**3/2 - (10*x**2)/3 + 11*x/2 +1

#plt.plot(x, ydata, 'o-')


fint = interp1d(x, ydata, kind='cubic')


print(fint(-1.5))


xc = np.linspace(0,5,101)
yc = fint(xc)

plt.plot(xc, yc, 'x-')
plt.plot(x, ydata, 's')
plt.plot(1.5, fint(1.5), 'r*')
#plt.show()