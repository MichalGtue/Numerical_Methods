import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set grid resolution
nx = 40
ny = 40

# Set old solution array
T = np.zeros((nx, ny))

# Set boundary conditions
T[0, :] = 40  # Left
T[nx-1, :] = 60  # Right
T[:, 0] = 20  # Bottom
T[:, ny-1] = 30
T[19:25, 19:25] = 100
fixedT = np.zeros((nx, ny))
fixedT[19:25, 19:25] = 1

# Set new solution array (including boundary conditions)
Tnew = T.copy()

# Required for plotting
x, y = np.meshgrid(np.arange(1, nx+1), np.arange(1, ny+1))

# Plotting 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, T, cmap='coolwarm', linewidth=0, antialiased=False)
title = ax.set_title("Iteration: " + "0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Temperature")
ax.set_zlim(0, np.max(T))
plt.show(block=False)

# Performing jacobi iterative procedure
for iter in range(1000):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            if fixedT[i, j] != 1:
                Tnew[i, j] = (T[i-1, j] + T[i+1, j] + T[i, j-1] + T[i, j+1]) / 4.0
    T = Tnew.copy()
    if iter % 10 == 0:
        surf.remove()
        surf = ax.plot_surface(x, y, T, cmap='coolwarm', linewidth=0, antialiased=False)
        title.set_text("Iteration: %i"%iter)
        fig.canvas.draw()
        fig.canvas.flush_events()