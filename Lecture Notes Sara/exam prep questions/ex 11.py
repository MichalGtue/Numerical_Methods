import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#fig = plt.figure()
#x = fig.add_subplot(111, projection= '3d')
# Make data.
#x = np.arange(-10, 10, 1000)
#y = np.arange(-10, 10, 1000)
#x,y = np.meshgrid(x, y)
#z = x**3+3*y -y**3-3*x

# Plot the surface.
#surf = ax.plot_surface(x,y,z,
#cmap=cm.magma,linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-10, 10)

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()


space = np.linspace(-10,10,100)
x,y = np.meshgrid(space, space)
z= x**3+3*y -y**3-3*x
fig = plt.figure()
ax=fig.add_subplot(111,projection= '3d')

surf=ax.plot_surface (x,y,z)
plt.show()
