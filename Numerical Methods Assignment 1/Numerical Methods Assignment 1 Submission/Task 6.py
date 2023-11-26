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
    if i == 99:
        pos_after_100_steps = pos       
    if i == 199:
        pos_after_200_steps = pos
    if i == 299:
        pos_after_300_steps = pos
    if i == 399:
        pos_after_400_steps = pos
    current_d = mean_square_displacement_f(pos) / (2* Number_Of_Dimensions * (i+1))
    dcoeff_vs_time[i, :] = [i + 1, current_d]
    line41.set_data(pos[:, 0], pos[:, 1])
    fig41.canvas.draw()
    fig41.canvas.flush_events()
    plt.pause(0.001)
#Also comment the lines above to hide the graph




#Prereq

#Task 6
#Again the naming, first number is the task number and the second is the number  of the graph

fig61 = plt.figure(figsize=(18,8))

ax61 = plt.subplot(2, 4, 1)
hist2d61 = ax61.hist2d(pos_after_100_steps[:, 0], pos_after_100_steps[:, 1], bins=15, cmap=cm.plasma)
ax61.set_title('Figure 6.1: Histogram for prisoner positions after 100 steps', size=7, weight='bold')
plt.colorbar(hist2d61[3], ax=ax61)


ax62 = plt.subplot(2, 4, 2)
hist2d62 = ax62.hist2d(pos_after_200_steps[:, 0], pos_after_200_steps[:, 1], bins=15, cmap=cm.plasma)
ax62.set_title('Figure 6.2: Histogram for prisoner positions after 200 steps', size=7, weight='bold')
plt.colorbar(hist2d62[3], ax=ax62)


ax63 = plt.subplot(2, 4, 5)
hist2d63 = ax63.hist2d(pos_after_300_steps[:, 0], pos_after_300_steps[:, 1], bins=15, cmap=cm.plasma)
ax63.set_title('Figure 6.3: Histogram for prisoner positions after 300 steps', size=7, weight='bold')
plt.colorbar(hist2d63[3], ax=ax63)


ax64 = plt.subplot(2, 4, 6)
hist2d64 = ax64.hist2d(pos_after_400_steps[:, 0], pos_after_400_steps[:, 1], bins=15, cmap=cm.plasma)
ax64.set_title('Figure 6.4: Histogram for prisoner positions after 400 steps', size=7, weight='bold')
plt.colorbar(hist2d64[3], ax=ax64)


ax65 = plt.subplot(1, 2, 2)
hist2d65 = ax65.hist2d(pos[:, 0], pos[:, 1], bins=15, cmap=cm.plasma)
ax65.set_title('Figure 6.5: Histogram for prisoner positions after the last step', size=12, weight='bold')
plt.colorbar(hist2d65[3], ax=ax65)



plt.show()