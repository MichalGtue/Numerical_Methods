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
z1 = 1/(4 * np.pi * expected_diffusion_coeff * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff * t*2))
z2 = 1/(4 * np.pi * expected_diffusion_coeff * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff * t*3))
z3 = 1/(4 * np.pi * expected_diffusion_coeff * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff * t*4))
z4 = 1/(4 * np.pi * expected_diffusion_coeff * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff * t*5))
fig = plt.figure()

fig = plt.figure()
ax71 = fig.add_subplot(2, 4, 1, projection='3d')
ax72 = fig.add_subplot(2, 4, 2, projection='3d')
ax73 = fig.add_subplot(2, 4, 5, projection='3d')
ax74 = fig.add_subplot(2, 4, 6, projection='3d')
ax75 = fig.add_subplot(1, 2, 2, projection='3d')

surf1 = ax71.plot_surface(x,y,z, cmap=cm.magma, linewidth=0, antialiased=0)
surf2 = ax72.plot_surface(x,y,z1, cmap=cm.magma, linewidth=0, antialiased=0)
surf3 = ax73.plot_surface(x,y,z2, cmap=cm.magma, linewidth=0, antialiased=0)
surf4 = ax74.plot_surface(x,y,z3, cmap=cm.magma, linewidth=0, antialiased=0)
surf5 = ax75.plot_surface(x,y,z4, cmap=cm.magma, linewidth=0, antialiased=0)




plt.show()
