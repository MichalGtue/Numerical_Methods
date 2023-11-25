#Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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

#For task 5

Number_Of_Dimensions = 2




# Prereqs


fig91, ax91 = plt.subplots()
line91, = ax91.plot([], [], 'o')
ax91.set_xlim(-20, 20)
ax91.set_ylim(-20,20)
#
boundry_condition = 12*np.cos(0.1*np.pi)
#
x1_for_9 = np.linspace(-12,12,10**4)
x2_for_9 = np.linspace(-12, boundry_condition, 10**4)
y1_for_9 = np.sqrt(12**2 - (x2_for_9**2))
y2_for_9 = -1* np.sqrt(12**2 - (x1_for_9**2))
ax91.plot(x2_for_9, y1_for_9, "r-")
ax91.plot(x1_for_9, y2_for_9, "r-")



escape_times = []
to_be_removed = []
Number_of_Prisoners = 5000
#Number_of_Steps = 1 # Since were running it until they leave we dont know how many steps it will take
pos=np.zeros([Number_of_Prisoners, 2])
step_number = 0

# This one is quicker but its vulnerable to the edge case as described in the assignment document figure 1

#Uncomment it to see it 
#while len(pos[:,0]) > 0:
#    step_number = step_number + 1
#    for p in range(len(pos[:, 0])):
#        if pos [p, 0] == 13 and pos[p,1] == 2:
#            to_be_removed.append(p)
#            escape_times.append(step_number-1)
#    pos = np.delete(pos, to_be_removed, axis=0)
#    to_be_removed = []
#    pos_ini = pos.copy()
#    pos = np.add(pos, new_step(len(pos[:,0])))
#    for n in range(len(pos[:,0])):
#        if np.linalg.norm(pos[n,:]) >= 12:
#            if pos[n, 1] >= 0 and boundry_condition <= pos[n, 0]:
#                pos[n,:] = [13,2]
#            else:
#               pos[n,:] = pos_ini[n,:]
#               new_maybe_correct_step = new_step(1)
#               while (np.linalg.norm(np.add(pos[n, :], new_maybe_correct_step)) >= 12 ):
#                   new_maybe_correct_step = new_step(1)
#               pos[n,:] = np.add(pos[n,:], new_maybe_correct_step)
#    line91.set_data(pos[:, 0], pos[:, 1])
#    fig91.canvas.draw()
#    fig91.canvas.flush_events()
#    plt.pause(0.0001)


# The one below solves the edge case but its a bit slower


#Check if correctly crosses the border
x_for_checking = sympy.symbols('x', positive=True)

boundry_condition_for_checking = 12*sympy.cos(0.1*sympy.pi)

f1 = sympy.Piecewise((sympy.sqrt(12**2 - x_for_checking**2), x_for_checking >= boundry_condition_for_checking), (0, True))



while len(pos[:,0]) > 0:
    step_number = step_number + 1
    for p in range(len(pos[:, 0])):  ## Check if any prisoner is in 13, 2 and remove it 
        if pos [p, 0] == 13 and pos[p,1] == 2:
            to_be_removed.append(p)
            escape_times.append(step_number-1)
    pos = np.delete(pos, to_be_removed, axis=0)
    to_be_removed = []
    pos_ini = pos.copy()
    pos = np.add(pos, new_step(len(pos[:,0])))
    for n in range(len(pos[:,0])):
        if np.linalg.norm(pos[n,:]) >= 12: # Check if they hit the boundry
            if pos[n,0] > 11:  # Can be removed but its much slower without
                slope_for_testing = (pos[n,1]-pos_ini[n,1])/(pos[n,0]-pos_ini[n,0])
                y_intercept_for_testing = pos[n,1] - (slope_for_testing)*pos[n,0]
                f2 = slope_for_testing * x_for_checking + y_intercept_for_testing  ## Draw a straight line between new point and initial point
                solution_to_be_tested = sympy.solve(f1 - f2, x_for_checking)
                if len(solution_to_be_tested)==1 and solution_to_be_tested[0] >= boundry_condition:
                   pos[n,:] = [13,2]      
                elif len(solution_to_be_tested) == 2 and solution_to_be_tested[1] >= boundry_condition:
                   pos[n,:] = [13,2]
                else:
                   pos[n,:] = pos_ini[n,:]
                   new_maybe_correct_step = new_step(1)
                   while (np.linalg.norm(np.add(pos[n, :], new_maybe_correct_step)) >= 12 ):
                       new_maybe_correct_step = new_step(1)
                   pos[n,:] = np.add(pos[n,:], new_maybe_correct_step)   
            else:
               pos[n,:] = pos_ini[n,:]
               new_maybe_correct_step = new_step(1)
               while (np.linalg.norm(np.add(pos[n, :], new_maybe_correct_step)) >= 12 ):
                   new_maybe_correct_step = new_step(1)
               pos[n,:] = np.add(pos[n,:], new_maybe_correct_step)
    line91.set_data(pos[:, 0], pos[:, 1])
    fig91.canvas.draw()
    fig91.canvas.flush_events()
    plt.pause(0.0001)



fig91, ax91 = plt.subplots(1,1)


ax91.hist(escape_times, bins=10)


plt.show()