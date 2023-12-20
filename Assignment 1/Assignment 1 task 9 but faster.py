#Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import sympy
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


step_size = 0.5
def new_step(N):
    pos = np.zeros([N, 2])
    rand_rad = get_random_radian(N) # To use the same direction for a pair of xy coordinates
    x_values = np.cos(rand_rad) * step_size
    y_values = np.sin(rand_rad) * step_size
    rand_array = np.column_stack([x_values, y_values])
    pos = np.add(pos, rand_array)
    return rand_array



Number_of_Prisoners = 1000

pos=np.zeros([Number_of_Prisoners, 2])
#for i in range(Steps_num):
#    pos = np.add(pos, rand_arr(500))





Task_3_x = pos[:,0]
Task_3_y = pos[:,1]
#Making the figure itself


fig, ax = plt.subplots()
line, = ax.plot([], [], 'o')
ax.set_xlim(-20, 20)
ax.set_ylim(-20,20)
#
boundry_condition = 12*np.cos(0.1*np.pi)
#
x1_for_8 = np.linspace(-12,12,10**4)
x2_for_8 = np.linspace(-12, boundry_condition, 10**4)
y1_for_8 = np.sqrt(12**2 - (x2_for_8**2))
y2_for_8 = -1* np.sqrt(12**2 - (x1_for_8**2))
ax.plot(x2_for_8, y1_for_8)
ax.plot(x1_for_8, y2_for_8)
#
#
#plt.show(block=False)



#Check if correctly crosses the border


escape_times = []



to_be_removed = []



Number_of_Prisoners = 100
Number_of_Steps = 1
pos=np.zeros([Number_of_Prisoners, 2])
step_number = 0


while len(pos[:,0]) > 0:
    step_number = step_number + 1
    for p in range(len(pos[:, 0])):
        if pos [p, 0] == 13 and pos[p,1] == 2:
            to_be_removed.append(p)
            escape_times.append(step_number-1)
    pos = np.delete(pos, to_be_removed, axis=0)
    to_be_removed = []
    pos_ini = pos.copy()
    pos = np.add(pos, new_step(len(pos[:,0])))
    for n in range(len(pos[:,0])):
        if np.linalg.norm(pos[n,:]) >= 12:
            if pos[n, 1] >= 0 and boundry_condition <= pos[n, 0]:
                plt.pause(3)
                pos[n,:] = [13,2]
            else:
               pos[n,:] = pos_ini[n,:]
               new_maybe_correct_step = new_step(1)
               while (np.linalg.norm(np.add(pos[n, :], new_maybe_correct_step)) >= 12 ):
                   new_maybe_correct_step = new_step(1)
               pos[n,:] = np.add(pos[n,:], new_maybe_correct_step)
    line.set_data(pos[:, 0], pos[:, 1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.00000000001)
##### Remove the above '#' to see the plots. I commented them out cause its makes the simulation much longer


fig91, ax91 = plt.subplots(1,1)

number_of_bins = 15

logbins = np.logspace(np.log10(np.min(escape_times)),np.log10(np.max(escape_times)),number_of_bins+1)
ax91.set_title('Figure 9.1: Animation for a random 500 step path of 1000 prisoners bounded by a fence with a gap from 0 to 0.1 radians', size=10, weight='bold')
ax91.set_xlabel('Prisoners position in x direction')
ax91.set_ylabel('Prisoners position in y direction')
ax91.hist(escape_times, bins=logbins)
plt.xscale('log')

plt.axvline(x = 1675.1462200196254, color = 'b', label = 'axvline - full height')

plt.show()