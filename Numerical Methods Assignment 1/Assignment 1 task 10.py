import numpy as np
import matplotlib.pyplot as plt
import time
import sympy

step_size = 0.5
radius = 12
Number_Of_Dimensions = 2

def mean_escape_time(r, s):
    '''Takes two input, first the radius second the step size and returns the mean escape time.'''
    epsilon = s
    diffusion_coeff_in_function = (epsilon**2)/(2 * Number_Of_Dimensions * 1) ## Delta t will always be one for our cases
    residence_time = (r**2)/(diffusion_coeff_in_function)  * (np.log10(epsilon**(-1)) + np.log10(2) + 8**(-1))
    return residence_time

print(mean_escape_time(12, 0.5))


print(np.array(range(1, 6)))