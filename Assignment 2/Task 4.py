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


#New for task 4, we need to define a new QMO
Q_MO_inputs = np.linspace(0,100, 1000)

#Empty lists to store the values
conc_lake_michigan_values = []
conc_lake_huron_values = []
conc_lake_erie_values = []
conc_lake_ontario_values = []



#Lake Superior is not included because its independent of the lake Michigan
for Q_MO in Q_MO_inputs: # Calculate the concentraions in the lakes at various bypass flowrates
    M = np.array([[Q_SH + k1*V_S,0,0,0,0], [0,Q_MH + k1*V_M + Q_MO,0,0,0], [-Q_SH, -Q_MH, Q_HE + k1*V_H, 0, 0], [0,0,-Q_HE, Q_EO + k1*V_E, 0], [0,-Q_MO,0,-Q_EO,Q_OO + k1*V_E]])
    sol_vec = np.array([S_in, S_in_M, S_in_H,S_in_E, S_in_O])
    conc_lake_michigan, conc_lake_huron, conc_lake_erie, conc_lake_ontario = np.linalg.solve(M,sol_vec)[1], np.linalg.solve(M,sol_vec)[2], np.linalg.solve(M,sol_vec)[3],np.linalg.solve(M, sol_vec)[4] #Yes its slower to calculate it 4 times but since it the matrix is so small it doesn't matter too much
    conc_lake_michigan_values.append(conc_lake_michigan)
    conc_lake_huron_values.append(conc_lake_huron)
    conc_lake_erie_values.append(conc_lake_erie)
    conc_lake_ontario_values.append(conc_lake_ontario)



#Making figures
fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot(2, 2, 1)
ax1.plot(Q_MO_inputs, conc_lake_michigan_values)
ax1.set_title("Figure 4.1: Concentration of PCB ([kg/yr]) in lake Michigan against bypass flowrates", size=8, weight='bold')
ax1.set_xlabel('Concentration of PCB ([kg/yr])')
ax1.set_ylabel('Bypass Flowrate [km**3/yr]')

ax2 = plt.subplot(2,2,2)
ax2.plot(Q_MO_inputs, conc_lake_huron_values)
ax2.set_title("Figure 4.2: Concentration of PCB ([kg/yr]) in lake Huron against bypass flowrates", size=8, weight='bold')
ax2.set_xlabel('Concentration of PCB ([kg/yr])')
ax2.set_ylabel('Bypass Flowrate [km**3/yr]')

ax3 = plt.subplot(2,2,3)
ax3.plot(Q_MO_inputs, conc_lake_erie_values)
ax3.set_title("Figure 4.3: Concentration of PCB ([kg/yr]) in lake Michigan against bypass flowrates", size=8, weight='bold')
ax3.set_xlabel('Concentration of PCB ([kg/yr])')
ax3.set_ylabel('Bypass Flowrate [km**3/yr]')

ax4 = plt.subplot(2,2,4)
ax4.plot(Q_MO_inputs, conc_lake_ontario_values)
ax4.set_title("Figure 4.4: Concentration of PCB ([kg/yr]) in lake Ontario against bypass flowrates", size=8, weight='bold')
ax4.set_xlabel('Concentration of PCB ([kg/yr])')
ax4.set_ylabel('Bypass Flowrate [km**3/yr]')


plt.show()

print(np.linalg.solve(M,sol_vec))