import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Taking the code from task 6
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
    dxdt[1] = (deltaH * -1*A * np.exp((-Ea)/(R*x[1])) * x[0]**3)/(rho*Cp)
    return dxdt


def midpoint_rule(fun,tspan, y0, number_of_points=100):
    '''Applies the midpoint rule on the system of two differential equations for task 7. \n
    fun = function 
    y0 = vector of initial conditions
    optional:\n
    number_of_points = how many steps. Default set to 100. Increasing this reduces error but increases computation time. '''
    dt = (tspan[1] - tspan[0])/number_of_points
    t = np.linspace(tspan[0], tspan[1], number_of_points+1)
    y = np.zeros((number_of_points+1, 2))
    y[0,0] = y0[0] ## Setting up the initial conditions
    y[0,1] = y0[1]
    for i in range(number_of_points):
        k1 = fun(t[i], y[i,:])
        k2 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k1)
        y[i+1,:] = y[i,:] + dt * k2
    return t, y


# Same plotting but now with the midpoint rule.
# Code copied from task 7


ini_cond_vec = [c0, T0] # Initial conditions a vector
tspan = [0,1000]

approx_sol = midpoint_rule(der_system_task6, tspan, ini_cond_vec)

solution = scipy.integrate.solve_ivp(der_system_task6, tspan, ini_cond_vec)

fig = plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.plot(solution.t, solution.y[0], label='Scipy solve_ivp')
ax1.plot(approx_sol[0], approx_sol[1][:,0], label='Midpoint rule')
ax1.set_ylabel('Concentration of A (mol/L)')
ax1.set_xlabel('Time (s)')
ax1.set_title('Concentration over time for component A')
ax1.legend()
ax2.plot(solution.t, solution.y[1],'r',  label='Scipy solve_ivp')
ax2.plot(approx_sol[0], approx_sol[1][:,1], label='Midpoint rule')
ax2.set_title('Temperature over time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Tempertature (K)')
fig.suptitle('Solution to the system of ODEs', weight='bold')
ax2.legend()
plt.show()