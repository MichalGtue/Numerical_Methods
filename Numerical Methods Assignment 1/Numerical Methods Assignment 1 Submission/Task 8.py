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



step_size = 0.5

pos = np.array([[0, 0]])

def get_xy_velocities(N):
    random_rad_function = get_random_radian(N)[0]
    random_x_y = np.array([np.cos(random_rad_function) * step_size, np.sin(random_rad_function) * step_size])
    return random_x_y


# Prereq from previous tasks

#Task 4 

def new_step(N):
    pos = np.zeros([N, 2])
    rand_rad = get_random_radian(N) # To use the same direction for a pair of xy coordinates
    x_values = np.cos(rand_rad) * step_size
    y_values = np.sin(rand_rad) * step_size
    rand_array = np.column_stack([x_values, y_values]) # Used to combine the x and y into seperate columns
    pos = np.add(pos, rand_array)
    return rand_array


Number_of_Steps = 500
Number_of_Prisoners = 1000

pos=np.zeros([Number_of_Prisoners, 2])


#Task 4 graph 1

#Uncomment the bottom lines if you want to see the graph of prisoners
#fig41, ax41 = plt.subplots()
#line41, = ax41.plot([], [], 'o')
#plt.show(block=False)
#ax41.set_xlim(-50, 50)
#ax41.set_ylim(-50,50)
#ax41.set_title('Path of 1000 prisoners after 500 steps')
#ax41.set_xlabel('x position')
#ax41.set_ylabel('y position')








#Prereq from previous tasks

def mean_square_displacement_f(j):
    displacement = np.array([])
    for i in range(len(j[:, 0])):
        displacement = np.append(displacement, np.linalg.norm(pos[i, :])**2)
    mean_square_displacement = np.mean(displacement)
    return mean_square_displacement

#For task 5
dcoeff_vs_time = np.zeros((Number_of_Steps, 2))
Number_Of_Dimensions = 2

#Prereqs

#Copied but modified from task 4
fig81, ax81 = plt.subplots()
line81, = ax81.plot([], [], 'o')

#Making it pretty
plt.show(block=False)
ax81.set_xlim(-20, 20)
ax81.set_ylim(-20,20)
ax81.set_title('Path of 1000 prisoners after 500 steps with bounds')
ax81.set_xlabel('x position')
ax81.set_ylabel('y position')





x1_for_8 = np.linspace(-12,12,10**4)
y1_for_8 = np.sqrt(12**2 - (x1_for_8**2))
y2_for_8 = -1* np.sqrt(12**2 - (x1_for_8**2))
ax81.plot(x1_for_8, y1_for_8, "r-")
ax81.plot(x1_for_8, y2_for_8, "r-")


plt.show(block=False)
pos=np.zeros([Number_of_Prisoners, 2])
pos = pos + 0.0

for i in range(Number_of_Steps):
    pos_ini = pos.copy()
    pos = np.add(pos, new_step(Number_of_Prisoners))
    for n in range(len(pos[:,0])):
        if np.linalg.norm(pos[n,:]) >= 12: # First check to see if the new position is outside the bounds
            pos[n,:] = pos_ini[n,:]     # Go back to initial position
            new_maybe_correct_step = new_step(1)
            while np.linalg.norm(np.add(pos[n,:], new_maybe_correct_step)) >= 12: # Check to see if new step is outside the bounds
                new_maybe_correct_step = new_step(1)
            pos[n,:] = np.add(pos[n,:], new_maybe_correct_step)
    line81.set_data(pos[:, 0], pos[:, 1])
    fig81.canvas.draw()
    fig81.canvas.flush_events()
    plt.pause(0.001)

# Make it so that the graph prints as a square so that its clear that the boundry is a circle and not an oval