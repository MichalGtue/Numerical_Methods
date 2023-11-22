#Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
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


Random_radian_big_number = 10**8 #Bigger exponent = more time

fig, ax = plt.subplots(1, 1)
ax.hist(get_random_radian(Random_radian_big_number), bins=30) 

#To make graph nicer
ax.set_title('Distribution of get_random_radian')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')

#####################################################################Print out for testing
#plt.show()

#Mean
Random_Radian_f_mean = np.mean(get_random_radian(Random_radian_big_number))

#Standard Deviatio
Random_Radian_f_stdev = np.std(get_random_radian(Random_radian_big_number))

print('Your mean with', Random_radian_big_number, 'is', Random_Radian_f_mean, 'and your standard deviation is', Random_Radian_f_stdev)

#Task 3

pos = np.array([[0, 0]])

def get_xy_velocities(N):
    random_rad_function = get_random_radian(N)[0]
    random_x_y = np.array([np.cos(random_rad_function) * 0.5, np.sin(random_rad_function) * 0.5])
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
ax31.set_title('Random Path of a single prisoner')
ax31.set_xlabel('Position in x')
ax31.set_ylabel('Position in y')


plt.show(block=False)
for i in range(1000):
    line.set_data(Task_3_x[i-20:i+1], Task_3_y[i-20:i+1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


###################################Remove for testing



#Task 4 

def new_step(N):
    pos = np.zeros([N, 2])
    rand_rad = get_random_radian(N) # To use the same direction for a pair of xy coordinates
    x_values = np.cos(rand_rad) * 0.5
    y_values = np.sin(rand_rad) * 0.5
    rand_array = np.column_stack([x_values, y_values]) # Used to combine the x and y into seperate columns
    pos = np.add(pos, rand_array)
    return rand_array


Number_of_Steps = 500
Number_of_Prisoners = 1000

pos=np.zeros([Number_of_Prisoners, 2])

#Task 4 graph 1
fig41, ax41 = plt.subplots()
line41, = ax41.plot([], [], 'o')

#Making it pretty
plt.show(block=False)
ax41.set_xlim(-50, 50)
ax41.set_ylim(-50,50)
ax41.set_title('Path of 1000 prisoners after 500 steps')
ax41.set_xlabel('x position')
ax41.set_ylabel('y position')



for i in range(Number_of_Steps):
    pos = np.add(pos, new_step(Number_of_Prisoners))
    line41.set_data(pos[:, 0], pos[:, 1])
    fig41.canvas.draw()
    fig41.canvas.flush_events()
    plt.pause(0.01)
## Takes a while to print out you have to be patient


#I assume that a 2d histogram is just a heatmeap


ffig42, ax42 = plt.subplots()


ax42.hist2d(pos[:, 0], pos[:, 1], bins=15, cmap=cm.plasma)


ax42.set_xlabel('Position in the x direction')
ax42.set_ylabel('Position in the y direction')
ax42.set_title('2D Histogram of the path taken by 500 prisoners after 1000 steps')



plt.show()