# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set grid resolution
nx = ny = 40

# Set old solution array
T = np.zeros((nx,ny))

# Set boundary conditions
T[0,:] = 40 # Left
T[nx-1,:] = 60 # Right
T[:,0] = 20 # Bottom
T[:,ny-1] = 30 # Top

# Set new solution array (including boundary conditions)
Tnew = T.copy()

# Create grid for plotting
x,y = np.meshgrid(np.arange(1,nx+1), np.arange(1,ny+1))

# Plot and update current solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x,y,Tnew,color="cornflowerblue")
title = ax.set_title("Iteration: " + "0")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Temperature')
plt.show(block=False) # Show but don't pause execution

# Perform iterations
for iter in range(1,1001):
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            Tnew[i,j] = (T[i-1,j]+T[i+1,j]+T[i,j-1]+T[i,j+1])/4.0 # Calculate new solution
    T = Tnew.copy()

    # Updating the plot 
    if iter%10 == 0:
        surf.remove() # Removing old surface
        title.set_text("Iteration: %i"%iter) # Changing the title 
        surf = ax.plot_surface(x,y,Tnew,cmap=cm.terrain,edgecolor='black',linewidth=0.2) # Making new surface
        fig.canvas.draw() # Drawing new figure 
        fig.canvas.flush_events() # Speeding up next rendering step
        
plt.show(block=True)
