#Libraries
import numpy as np
import timeit
import time
from scipy.linalg import lu 

#task 2 matrix
#Define the values of the Q's (given in km^-3 / yr)
Q_SH = 72
Q_MH = 38
Q_HE = 160
Q_EO = 185
Q_OO = 215

#Define values for the different PCB source S_in's (given in kg/yr)

S_in   = 180
S_in_M = 810
S_in_H = 630
S_in_E = 2750
S_in_O = 3820

#Generating our coefficient matrix
M = np.array([[Q_SH,0,0,0,0], [0,Q_MH,0,0,0], [-Q_SH, -Q_MH, Q_HE, 0, 0], [0,0,-Q_HE, Q_EO, 0], [0,0,0,-Q_EO,Q_OO]])
#print(M) #Uncomment to see the coefficient matrix


k1= 0.00584

def sunlight_break_down():
    
    # first order function
    r=k1*Cx
        