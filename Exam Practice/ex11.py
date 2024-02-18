from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

x = np.arange(-2,2,0.025)
y = np.arange(-2,2,0.025)
x,y = np.meshgrid(x,y)

z = x**3 + 3*y - y**3 - 3*x

surf = ax.plot_surface(x,y,z, cmap=cm.magma, linewidth=0, antialiased=False)


plt.show()