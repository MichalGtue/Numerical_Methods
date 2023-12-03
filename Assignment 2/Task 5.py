#Libraries
import numpy as np
import matplotlib.pyplot as plt

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

#Defining k2
k2 = 2 * 10**7

#Volume of tank is given in m^3 but everything else is in km^3. Thus, the volume is given in km^3
V_tank = 1*10**-9


# Not given in the question so taken as 100
Q_MO = 100



## For PFR Graph
conc_in_tank = [12.15924102] # Set to the initial concentration in the lake
for i in range(1000):
    new_conc = (conc_in_tank[-1] * Q_MO)/ (Q_MO+V_tank*k2) 
    conc_in_tank.append(new_conc)
conc_in_tank.pop(0) # Removing the initial concentration
print(len(conc_in_tank))




M = np.array([[Q_SH + k1*V_S,0,0,0,0], [0,Q_MH + k1*V_M + Q_MO,0,0,0], [-Q_SH, -Q_MH, Q_HE + k1*V_H, 0, 0], [0,0,-Q_HE, Q_EO + k1*V_E, 0], [0,-Q_MO,0,-Q_EO,Q_OO + k1*V_E]])
sol_vec = np.array([S_in, S_in_M, S_in_H,S_in_E, S_in_O])