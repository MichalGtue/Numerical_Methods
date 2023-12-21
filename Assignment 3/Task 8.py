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

# For the challenge the master function is provided below. The default method is set to the rk2 method.
# Check the documentation for the supported methods
def master_function(fun,tspan, y0, method='rk2', number_of_points=100):
    '''General function to solve system of differential equations. Does not work on single differential equations. \n
    fun = function 
    y0 = vector of initial conditions
    optional:\n
    method = You can select the method with which your system of differential equations will be evaluated. Default set to second order Runge-Kutta. \n
    Supported methods : midpoint method ('midpoint'), euler method ('euler'), Classical second order Runge-Kutta ('rk2'), classical fourth order Runge-Kutta ('rk4').
    number_of_points = how many steps. Default set to 100. Increasing this reduces error but increases computation time. '''
    dt = (tspan[1] - tspan[0])/number_of_points
    t = np.linspace(tspan[0], tspan[1], number_of_points+1)
    y = np.zeros((number_of_points+1, len(y0))) # len(y0) because you would need an initial condition for each derivative.
    for i in range(len(y0)): #initial conditions as a loop to ensure universability.
        y[0,i] = y0[i]
    if method == 'midpoint':
        for i in range(number_of_points):
            k1 = fun(t[i], y[i,:])
            k2 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k1)
            y[i+1,:] = y[i,:] + dt * k2
    elif method == 'euler':
        for i in range(number_of_points):
            y[i+1,:] = y[i,:] + dt * fun(t[i], y[i,:])
    elif method == 'rk2':
        for i in range(number_of_points):
            k1 = fun(t[i], y[i,:])
            k2 = fun(t[i] + dt, y[i] + dt*k1)
            y[i+1,:] = y[i] + dt*0.5*(k1+k2)
    elif method == 'rk4':
        for i in range(number_of_points):
            k1 = fun(t[i], y[i,:])
            k2 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k1)
            k3 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k2)
            k4 = fun(t[i] +dt, y[i,:] + dt*k3)
            y[i+1,:] = y[i] + dt*((1/6)*k1 + (1/3)*(k2+k3) + (1/6)*k4)
    else:
        return 'Unknown method specified. Check documentation for supported methods' # In case an unknown method is specified
    return t, y

# Same plotting but now with the midpoint rule.
# Code copied from task 7



ini_cond_vec = [c0, T0] # Initial conditions a vector
tspan = [0,1000]


approx_sol = master_function(der_system_task6, tspan, ini_cond_vec, 'midpoint')

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