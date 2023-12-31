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



def new_step(N):
    pos = np.zeros([N, 2])
    rand_rad = get_random_radian(N) # To use the same direction for a pair of xy coordinates
    x_values = np.cos(rand_rad) * 0.5
    y_values = np.sin(rand_rad) * 0.5
    rand_array = np.column_stack([x_values, y_values])
    pos = np.add(pos, rand_array)
    return rand_array


Steps_num = 500
Number_of_Prisoners = 1000

pos=np.zeros([Number_of_Prisoners, 2])
#for i in range(Steps_num):
#    pos = np.add(pos, rand_arr(500))

print(pos[:,0])



Task_3_x = pos[:,0]
Task_3_y = pos[:,1]
#Making the figure itself


fig, ax = plt.subplots()
line, = ax.plot([], [], 'o')
ax.set_xlim(-20, 20)
ax.set_ylim(-20,20)


# Show the plot without blocking
plt.show(block=False)

for i in range(Steps_num):
    pos = np.add(pos, new_step(Number_of_Prisoners))
    line.set_data(pos[:, 0], pos[:, 1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)


from matplotlib import cm
fig42, ax42 = plt.subplots()


ax42.hist2d(pos[:, 0], pos[:, 1], bins=15, cmap=cm.plasma)


ax42.set_xlabel('Position in the x direction')
ax42.set_ylabel('Position in the y direction')
ax42.set_title('2D Histogram of the path taken by 500 prisoners after 1000 steps')

ax42.set_xlim(-50, 50)
ax42.set_ylim(-50, 50)

plt.show()

## test
print('test')