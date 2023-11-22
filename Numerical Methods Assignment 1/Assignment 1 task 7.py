from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
expected_diffusion_coeff = 0.0625

t=100

x = np.arange(-15, 15, 0.005)
y = np.arange(-15, 15, 0.005)

x,y = np.meshgrid(x, y)

z = 1/(4 * np.pi * expected_diffusion_coeff * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff * t))
fig = plt.figure()

#for i in range(3):
#    axi = fig.supplot(2, i+1, i+1, projection='3d')
#    suri = ax.plot_surface(x,y,z, cmap=cm.magma, linewidth=0, antialiased=0)

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x,y,z, cmap=cm.magma, linewidth=0, antialiased=0)

plt.show()
