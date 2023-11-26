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

#Comment out the lines below to hide the graph
fig41, ax41 = plt.subplots()
line41, = ax41.plot([], [], 'o')
plt.show(block=False)
ax41.set_xlim(-50, 50)
ax41.set_ylim(-50,50)
ax41.set_title('Path of 1000 prisoners after 500 steps')
ax41.set_xlabel('x position')
ax41.set_ylabel('y position')








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


for i in range(Number_of_Steps):
    pos = np.add(pos, new_step(Number_of_Prisoners))
    current_d = mean_square_displacement_f(pos) / (2* Number_Of_Dimensions * (i+1))
    dcoeff_vs_time[i, :] = [i + 1, current_d]
    line41.set_data(pos[:, 0], pos[:, 1])
    fig41.canvas.draw()
    fig41.canvas.flush_events()
    plt.pause(0.001)
#Also comment the lines above to hide the graph


x_values_for_task_5 = np.linspace(0, 500, 100)
expected_diffusion_coeff = (step_size**2)/(2*Number_Of_Dimensions* 1) + 0*x_values_for_task_5

fig51, ax51 = plt.subplots()
ax51.scatter(dcoeff_vs_time[:,0], dcoeff_vs_time[:,1])
ax51.plot(x_values_for_task_5, expected_diffusion_coeff, label='Expected Diffusion Coefficient')
#Make the label for the straight line show


#Making ax51 pretty
ax51.set_xlabel('Number of Steps')
ax51.set_ylabel('Diffusion Coefficient')
ax51.set_title('Expected Diffusion Coefficient vs number of steps')



plt.show()
