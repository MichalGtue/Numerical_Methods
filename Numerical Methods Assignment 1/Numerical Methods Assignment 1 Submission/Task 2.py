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


Random_radian_big_number = 10**8 #Bigger exponent = more time

fig, ax21 = plt.subplots(1, 1)
ax21.hist(get_random_radian(Random_radian_big_number), bins=50) 

#To make graph nicer
ax21.set_title('Figure 2.1: Histogram for the distribution of function "get_random_radian" for N random radians', size=12, weight='bold')
ax21.set_xlabel('Radian value (0, 2π)')
ax21.set_ylabel('Frequency of random radian value')
ax21.set_ylim([1.5*10**6,2.5*10**6])

plt.show()
#Mean
Random_Radian_f_mean = np.mean(get_random_radian(Random_radian_big_number))

#Standard Deviatio
Random_Radian_f_stdev = np.std(get_random_radian(Random_radian_big_number))

print('The randon radian array', Random_radian_big_number, ' has a median of', Random_Radian_f_mean, 'and its standard deviation is', Random_Radian_f_stdev, '.')
