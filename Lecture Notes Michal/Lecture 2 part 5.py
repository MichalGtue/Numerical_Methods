from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-2, 2, 0.005)
y = np.arange(-2,2, 0.005)

x,y = np.meshgrid(x, y)

z= x* y * np.exp(-x**2 - y**2)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x,y,z, cmap=cm.magma, linewidth=0, antialiased=0)

plt.show()
