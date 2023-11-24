import numpy as np
import matplotlib.pyplot as plt
import time
import sympy



diffusion_coeff = 0.0625
radius = 12
epsilon = 0.5


Output = (radius**2)/(diffusion_coeff)  * (np.log10(epsilon**(-1)) + np.log10(2) + 8**(-1))

print(Output)