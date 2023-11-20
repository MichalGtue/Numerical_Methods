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



###################################Remove for testing


#Task 4 



def rand_arr(N):
    pos = np.zeros([N, 2])
    rand_rad = get_random_radian(N) # To use the same direction for a pair of xy coordinates
    x_values = np.cos(rand_rad) * 0.5
    y_values = np.sin(rand_rad) * 0.5
    rand_array = np.column_stack([x_values, y_values])
    pos = np.add(pos, rand_array)
    return rand_array


Steps_num = 10

pos=np.zeros([500, 2])
for i in range(Steps_num):
    pos = np.add(pos, rand_arr(500))

print(pos)

Task_3_x = pos[:,0]
Task_3_y = pos[:,1]

#Making the figure itself
fig31 = plt.figure(figsize=(8,7))
ax31 = plt.subplot(1, 1, 1)
line, = ax31.plot(Task_3_x, Task_3_y)

print(pos)





#print(rand_arr(100)) # Gives N amount of pairs of xy coordinates

#print(rand_arr(6))

#print(pos.shape)