import numpy as np
import matplotlib.pyplot as plt

x_values = [2,4,4.5,5.8]
y_values = [4,4,9,-12]
p = np.array([x_values,y_values]).T

p3= np.polyfit(p[:,0],p[:,1],3)

xh = np.linspace(1.5,6,1000)
yh = np.polyval(p3, xh)

plt.plot(xh,yh)
plt.plot(p[:,0],p[:,1], 'o',label='data')
plt.show()
