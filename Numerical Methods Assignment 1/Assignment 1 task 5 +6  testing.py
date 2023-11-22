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


# Show the plot without blocking
plt.show(block=False)

for i in range(Number_of_Steps):
    pos = np.add(pos, new_step(Number_of_Prisoners))
    line.set_data(pos[:, 0], pos[:, 1])



from matplotlib import cm
fig42, ax42 = plt.subplots()




#First to get mean square displacement









def mean_square_displacement_f(j):
    displacement = np.array([])
    for i in range(len(j[:, 0])):
        displacement = np.append(displacement, np.linalg.norm(pos[i, :])**2)
    mean_square_displacement = np.mean(displacement)
    return mean_square_displacement




#print(mean_square_displacement_f(pos))

dcoeff_vs_time = np.zeros([Number_of_Steps,2])



dcoeff_vs_time = np.zeros((Number_of_Steps, 2))
pos = np.zeros([Number_of_Prisoners, 2])
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


blank_2x2_array = np.zeros((1,2))




x_values_for_task_5 = np.linspace(0, 500, 100)
expected_diffusion_coeff = (step_size**2)/(2*Number_Of_Dimensions* 1) + 0*x_values_for_task_5



print(dcoeff_vs_time)
fig51, ax51 = plt.subplots()
ax51.scatter(dcoeff_vs_time[:,0], dcoeff_vs_time[:,1])
ax51.plot(x_values_for_task_5, expected_diffusion_coeff)




fig61 = plt.figure(figsize=(15,8))

ax61 = plt.subplot(2, 4, 1)
ax61.hist2d(pos_after_100_steps[:, 0], pos_after_100_steps[:, 1], bins=15, cmap=cm.plasma)
ax61.set_title('Histogram after 100 steps')

ax62 = plt.subplot(2, 4, 2)
ax62.hist2d(pos_after_200_steps[:, 0], pos_after_200_steps[:, 1], bins=15, cmap=cm.plasma)
ax62.set_title('Histogram after 200 steps')

ax63 = plt.subplot(2, 4, 5)
ax63.hist2d(pos_after_300_steps[:, 0], pos_after_300_steps[:, 1], bins=15, cmap=cm.plasma)
ax63.set_title('Histogram after 300 steps')

ax64 = plt.subplot(2, 4, 6)
ax64.hist2d(pos_after_400_steps[:, 0], pos_after_400_steps[:, 1], bins=15, cmap=cm.plasma)
ax64.set_title('Histogram after 400 steps')

ax65 = plt.subplot(1, 2, 2)
ax65.hist2d(pos[:, 0], pos[:, 1], bins=15, cmap=cm.plasma)
ax65.set_title('Histogram after the last step')
plt.show()
