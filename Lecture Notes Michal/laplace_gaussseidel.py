import numpy as np
import matplotlib.pyplot as plt

# Set grid resolution
nx = 40
ny = 40

# Set old solution array
T = np.zeros((nx,ny))

# Set boundary conditions
T[0,:] = 40 # Left
T[nx-1,:] = 60 # Right
T[:,0] = 20 # Bottom
T[:,ny-1] = 30

# Required for plotting
x, y = np.meshgrid(np.arange(1,nx+1), np.arange(1,ny+1))
plt.figure()
plt.pcolormesh(x,y,T)
plt.colorbar()
for iter in range(1000):
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            # Explicit 
            T[i,j] = (T[i-1,j]+T[i+1,j]+T[i,j-1]+T[i,j+1])/4.0
    
    plt.title('Iteration: ' + str(iter+1))
    plt.pcolormesh(x,y,T)
    plt.axis([0,nx,0,ny])
    plt.draw()
    plt.pause(0.00001)

plt.show()
