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


for i in range(Number_of_Steps):
    pos = np.add(pos, new_step(Number_of_Prisoners))
    if i == 99:
        pos_after_100_steps = pos       #Kinda ugly but it works
    if i == 199:
        pos_after_200_steps = pos
    if i == 299:
        pos_after_300_steps = pos
    if i == 399:
        pos_after_400_steps = pos
    current_d = mean_square_displacement_f(pos) / (2* Number_Of_Dimensions * (i+1))
    dcoeff_vs_time[i, :] = [i + 1, current_d]
#    line41.set_data(pos[:, 0], pos[:, 1])
#    fig41.canvas.draw()
#    fig41.canvas.flush_events()
#    plt.pause(0.001)
#Also uncomment the lines above to see 



#Prereqs

t=100

x = np.arange(-15, 15, 0.005)
y = np.arange(-15, 15, 0.005)

x,y = np.meshgrid(x, y)



expected_diffusion_coeff_numerically = 0.0625

z = 1/(4 * np.pi * expected_diffusion_coeff_numerically * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff_numerically * t))
z1 = 1/(4 * np.pi * expected_diffusion_coeff_numerically * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff_numerically * t*2))
z2 = 1/(4 * np.pi * expected_diffusion_coeff_numerically * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff_numerically * t*3))
z3 = 1/(4 * np.pi * expected_diffusion_coeff_numerically * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff_numerically * t*4))
z4 = 1/(4 * np.pi * expected_diffusion_coeff_numerically * t) *  np.exp(-(x**2 + y**2)/(4 * expected_diffusion_coeff_numerically * t*5))


fig71 = plt.figure(figsize=(18,8))
ax71 = fig71.add_subplot(2, 4, 1, projection='3d')
ax71.set_title('Figure 7.1: Probability density function 3D projection at 100 steps', size=6, weight='bold')
ax71.set_xlabel('x')
ax71.set_ylabel('y')
ax71.set_zlabel('pdf')
ax72 = fig71.add_subplot(2, 4, 2, projection='3d')
ax72.set_title('Figure 7.2: Probability density function 3D projection at 200 steps', size=6, weight='bold')
ax72.set_xlabel('x')
ax72.set_ylabel('y')
ax72.set_zlabel('pdf')
ax73 = fig71.add_subplot(2, 4, 5, projection='3d')
ax73.set_title('Figure 7.3: Probability density function 3D projection at 300 steps', size=6, weight='bold')
ax73.set_xlabel('x')
ax73.set_ylabel('y')
ax73.set_zlabel('pdf')
ax74 = fig71.add_subplot(2, 4, 6, projection='3d')
ax74.set_title('Figure 7.4: Probability density function 3D projection at 400 steps', size=6, weight='bold')
ax74.set_xlabel('x')
ax74.set_ylabel('y')
ax74.set_zlabel('pdf')
ax75 = fig71.add_subplot(1, 2, 2, projection='3d')
ax75.set_title('Figure 7.5: Probability density function 3D projection at 500 steps', size=10, weight='bold')
ax75.set_xlabel('x')
ax75.set_ylabel('y')
ax75.set_zlabel('pdf')

surf1 = ax71.plot_surface(x,y,z, cmap=cm.magma, linewidth=0, antialiased=0)
surf2 = ax72.plot_surface(x,y,z1, cmap=cm.magma, linewidth=0, antialiased=0)
surf3 = ax73.plot_surface(x,y,z2, cmap=cm.magma, linewidth=0, antialiased=0)
surf4 = ax74.plot_surface(x,y,z3, cmap=cm.magma, linewidth=0, antialiased=0)
surf5 = ax75.plot_surface(x,y,z4, cmap=cm.magma, linewidth=0, antialiased=0)


plt.show()