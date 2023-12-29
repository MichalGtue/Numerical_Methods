import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

#Initialize the variables
A = 3.1
rho = 1.25 # kg * m^-3
Cp = 1200 # J * kg^-1 * K^-1
deltaH = -18400 # we need to keep units constant thus J * mol^-1
Ea = 25800 # Same logic as above because 8.3145 will be used for R thus J * mol^-1
c0 = 10
T0 = 313
R = 8.3145 


def der_system_task6(t, x):
    '''Solves the following system of equations: \n
    dCadt = -A*exp(-Ea/RT) * Ca^3 \n
    dTdt = (deltaH * -A*exp(-Ea/RT) * Ca^3)/(rho*Cp) \n
    t =  vector for time span \n
    x = vector where first position is the concentration and second is the temperature \n'''
    dxdt = np.zeros(2) # two equations
    dxdt[0] = -1*A * np.exp((-Ea)/(R*x[1])) * x[0]**3 ## Represents dcadt
    dxdt[1] = (deltaH * -1*A * np.exp((-Ea)/(R*x[1])) * x[0]**3)/(rho*Cp) #Represents dTdt
    return dxdt

ini_cond_vec = [c0, T0] # Initial conditions a vector
tspan = [0,1000]
solution = scipy.integrate.solve_ivp(der_system_task6, tspan, ini_cond_vec)

fig = plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.plot(solution.t, solution.y[0])
ax2.plot(solution.t, solution.y[1])
ax1.set_ylabel('Concentration of A (mol/L)')
ax1.set_xlabel('Time (s)')
ax1.set_title('Concentration over time for component A')
ax2.set_title('Temperature over time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Tempertature (K)')
fig.suptitle('Solution to the system of ODEs', weight='bold')

plt.show()