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


def first_order_euler_system(fun,tspan, y0, number_of_points=100):
    dt = (tspan[1] - tspan[0])/number_of_points
    t = np.linspace(tspan[0], tspan[1], number_of_points+1)
    y = np.zeros(number_of_points+1)
    y[0] = y0 ## Setting up the initial conditions
    for i in range(number_of_points):
        y[i+1] = y[i] + dt * fun(t[i], y[i])
    return t, y