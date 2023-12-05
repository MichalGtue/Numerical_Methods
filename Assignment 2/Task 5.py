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
Q_MO = 20



## For PFR Graph
# Making the matrix

PFR_matrix = np.zeros((1005,1005))
new_sol_vec = np.zeros(1005)

M = np.array([[Q_SH + k1*V_S,0,0,0,0], [0,Q_MH + k1*V_M + Q_MO,0,0,0], [-Q_SH, -Q_MH, Q_HE + k1*V_H, 0, 0], [0,0,-Q_HE, Q_EO + k1*V_E, 0], [0, 0,0,-Q_EO,Q_OO + k1*V_E]])
PFR_matrix[:5, :5] = M # Update the new matrix 
PFR_matrix[4, -1] = -Q_MO # Inflow from the last PFR
sol_vec = np.array([S_in, S_in_M, S_in_H,S_in_E, S_in_O])
new_sol_vec[:5] = sol_vec

PFR_matrix[5, 1] = Q_MO # First PFR is dependent on the concentration of lake Michigan
PFR_matrix[5, 5] = -(Q_MO + V_tank*k2)

for i in range(999):
    PFR_matrix[6+i, 5+i] = Q_MO 
    PFR_matrix[6+i, 6+i] = -(Q_MO + V_tank*k2)

solution = np.linalg.solve(PFR_matrix, new_sol_vec)
print(solution[0:5]) # Prints the concentraion of the PCBs in the lakes


x = np.linspace(1, 1000, 1000)
ax1 = plt.subplot(1,1,1)
ax1.plot(x, np.linalg.solve(PFR_matrix,new_sol_vec)[5:])
ax1.set_title("Figure 5.1: Concentration profile of PCB in the PFR", size=11, weight='bold')
ax1.set_xlabel('PFR Tank Number')
ax1.set_ylabel('Concentration of PCB ([kg/yr]')

plt.show()