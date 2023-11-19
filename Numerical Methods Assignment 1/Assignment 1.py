#Libraries
import numpy as np
import matplotlib.pyplot as plt

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

#Print out for testing
#plt.show()

#Mean
Random_Radian_f_mean = np.mean(get_random_radian(Random_radian_big_number))

#Standard Deviatio
Random_Radian_f_stdev = np.std(get_random_radian(Random_radian_big_number))

print('Your mean with', Random_radian_big_number, 'is', Random_Radian_f_mean, 'and your standard deviation is', Random_Radian_f_stdev)

#Task 3
