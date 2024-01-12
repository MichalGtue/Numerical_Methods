# task 7
import numpy as np
import matplotlib.pyplot as plt

def get_random_radians(N):
    # Generate an array of N random numbers between 0 and 2π radians
    return np.random.rand(N) * 2 * np.pi

def get_xy_velocities(N): # HERE THE N is the number of prisonors
    x = 0.5 * np.cos(get_random_radians(N)) 
    y = 0.5 * np.sin(get_random_radians(N)) 
    return x, y


fig = plt.figure(figsize=(12, 5))
ax_surface = fig.add_subplot(121, projection='3d')
hist_gram = fig.add_subplot(122)


def next_position(x_now, y_now, N): # i need to add the N (Pass N as a parameter)
    x1, y1 = get_xy_velocities(N)
    new_x = x_now + x1
    new_y = y_now + y1
    return new_x, new_y


Np=500


import numpy as np
import matplotlib.pyplot as plt

# set up Green's function using the parameters given
def GreenFunc(x, y, t, D):
    if t == 0:
        return 0  # avoiding division by zero by returning 0 in each case
    else:
        p1 = 1 / (4 * np.pi * D * t)
        p2 = -((x**2 + y**2) / (4 * D * t))  
        pdf = p1 * np.exp(p2)
        return pdf
x_position = np.zeros(Np)
y_position = np.zeros(Np)


# make a function for the circle
def inside_circle(x_coord, y_coord, radius):
    return x_coord**2 + y_coord**2 <= radius**2

# define constants
Np = 1000
num_steps = 500
radius = 12

# make figure 
fig, ax = plt.subplots()
scatter = ax.scatter(x_position, y_position)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
from matplotlib.patches import Circle
# Plot the circle
circle = Circle((0, 0), radius, color='r', fill=False)
ax.add_patch(circle)
Np = 5
# open empty list for positions
final_x_position_list = []
final_y_position_list = []
x_position = np.zeros(Np)
y_position = np.zeros(Np)

# give positions of prisoners after each step
for i in range(num_steps):
    x_position, y_position = next_position(x_position, y_position, Np)
    # if prisoners are not inside boundary, assign previous step
    for j in range(Np):
        while not inside_circle(x_position[j], y_position[j], radius):
            x_position[j], y_position[j] = next_position(x_position[j-1], y_position[j-1], 1)  # Use N=1 to update the position for one prisoner

    # append the positions of the prisoners
    final_x_position_list.append(x_position)
    final_y_position_list.append(y_position)
    #combine positions of prisoners into one column
    scatter.set_offsets(np.column_stack((x_position, y_position)))
    # animate graph over time to confirm boundary
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.title(f'Time Step: {i}')
    plt.pause(0.01)


plt.show()


num_steps_list = [5, 100, 500]
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

# combine x and y coordinates into distance
def distance(x, y):
    power =np.power(x,2) + np.power(y,2)
    d_cal =np.sqrt(power)
    return d_cal

# make loop for different positions per number in num_steps_list
for steps, i in enumerate(num_steps_list):
    positions_distance = distance(final_x_position_list[i - 1], final_y_position_list[i - 1])

    # Convert these positions into one list
    positions = np.array(positions_distance).flatten().tolist()

    # Create a histogram for the distances at each specified number of steps
    hist = axs[steps].hist(positions, bins=20, edgecolor='black')
    axs[steps].set_title(f'distances from origin at {i} steps')
    axs[steps].set_xlabel('distance from origin')
    axs[steps].set_ylabel('number of prisoners')

plt.show()

