#Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
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



#Task 3
#Task 4

pos = np.zeros([5, 2])

Number_of_Steps = 500
def get_xy_velocities(N):
    random_rad_function = get_random_radian(N)
    random_x_y = np.array([np.cos(random_rad_function) * 0.5, np.sin(random_rad_function) * 0.5])
    return random_x_y

#for N in range(1, 1000):  
    Current_Step = get_xy_velocities(N)
    pos = np.vstack((pos, pos[-1, :] + Current_Step)) # New row = old row plus new step

#print(pos + get_xy_velocities(5))
#Extracting x and y coordinates

#Task_3_x = pos[:,0]
#Task_3_y = pos[:,1]

#Making the figure itself
fig31 = plt.figure(figsize=(8,7))
#ax31 = plt.subplot(1, 1, 1)
#line, = ax31.plot(Task_3_x, Task_3_y)
#
##Making the figure pretty
#ax31.set_title('Random Path of a single Prispner')
#ax31.set_xlabel('Position in x')
#ax31.set_ylabel('Position in y')
#
#
#
#for i in range(1000):
#    line.set_data(Task_3_x[i-20:i+1], Task_3_y[i-20:i+1])
#    fig31.canvas.draw()
#    fig31.canvas.flush_events()
#    plt.pause(0.001)


###################################Remove for testing


#Task 4 

def rand_arr(N):
    rand_rad = get_random_radian(N)
    return rand_rad

print('Hello')