#Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# Task 1
#You cannot change the range using numpy.random.rand() will always give you a number from 0 to 1.
#We can just multiply it by 2pi 

Random_N_2pi = 2 * np.pi * np.random.rand()

# print(Random_N_2pi) # <- For testing

#Task 2
#From numpy.org Parameters:

#    d0, d1, …, dnint, optional The dimensions of the returned array, must be non-negative. If no argument is given a single Python float is returned.

#Add documentation and maybe add case where if not an integer give error
def get_random_radian(N):
    radian_array = Random_N_2pi = 2 * np.pi * np.random.rand(N)
    return radian_array

# Prereq from previous tasks

step_size = 0.5

pos = np.array([[0, 0]])

def get_xy_velocities(N):
    random_rad_function = get_random_radian(N)[0]
    random_x_y = np.array([np.cos(random_rad_function) * step_size, np.sin(random_rad_function) * step_size])
    return random_x_y

#Check if magnitude = 0.5
#print(np.linalg.norm(get_xy_velocities(0))) #Norm does pythagoras

#Here range is the number of steps
#I found vstack which adds the new step to the position array as a new row
for N in range(1, 1000):  
    Current_Step = get_xy_velocities(N)
    pos = np.vstack((pos, pos[-1, :] + Current_Step)) # New row = old row plus new step

#Extracting x and y coordinates

Task_3_x = pos[:,0]
Task_3_y = pos[:,1]

#Making the figure itself
fig31 = plt.figure(figsize=(8,7))
ax31 = plt.subplot(1, 1, 1)
line, = ax31.plot(Task_3_x, Task_3_y)

#Making the figure pretty
ax31.set_title('Figure 3.1: Random 1000 step path of a single prisoner animated simulation', size=12, weight='bold')
ax31.set_xlabel('Prisoner position in x')
ax31.set_ylabel('Prisoner position in y')

plt.show(block=False)
for i in range(1000):
    line.set_data(Task_3_x[i-20:i+1], Task_3_y[i-20:i+1])
    fig31.canvas.draw()
    fig31.canvas.flush_events()
    plt.pause(0.0001)