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

#Volumes (km^3)

V_S = 12000
V_M = 4900
V_H = 3500
V_E = 480
V_O = 1640

#Defining k1
k1 = 0.00584


#Generating our coefficient matrix
M = np.array([[Q_SH + k1*V_S,0,0,0,0], [0,Q_MH + k1*V_M,0,0,0], [-Q_SH, -Q_MH, Q_HE + k1*V_H, 0, 0], [0,0,-Q_HE, Q_EO + k1*V_E, 0], [0,0,0,-Q_EO,Q_OO + k1*V_E]])
#print(M) #Uncomment to see the coefficient matrix


#Solution Vector
sol_vec = np.array([S_in, S_in_M, S_in_H,S_in_E, S_in_O])


print(np.linalg.solve(M, sol_vec))
