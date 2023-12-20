import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Taking code from task 10
# Needed later

l1_value = 1
l2_value = 1
m1_value = 1
m2_value = 1

def der_system_task10(t, x, m1=m1_value, m2=m2_value, l1=l1_value, l2=l2_value):
    '''Solves the following system of equations: \n
    dCadt = -A*exp(-Ea/RT) * Ca^3 \n
    dTdt = (deltaH * -A*exp(-Ea/RT) * Ca^3)/(rho*Cp) \n
    t =  vector for time span \n
    x = vector (theta1, omega1, theta2, omega2) \n'''
    dxdt = np.zeros(4) # two equations
    delta = x[2] - x[0]
    dxdt[0] = x[1]
    dxdt[1] = (m2 * l1 * (x[1]**2) * np.sin(delta)*np.cos(delta) + m2 * 9.81*np.sin(x[2])*np.cos(delta) + m2*l2*(x[3]**2) * np.sin(delta) - (m1+m2)*9.81*np.sin(x[0]))/((m1+m2)*l2 - m2*l2*(np.cos(delta))**2)
    dxdt[2] = x[3]
    dxdt[3] = (-m2*l2*(x[3]**2)*np.sin(delta)*np.cos(delta) + (m1+m2)*(9.81*np.sin(x[0])*np.cos(delta) - l1 * (x[1]**2) * np.sin(delta) - 9.81 * np.sin(x[2])))/((m1+m2)*l2 - m2*l2*(np.cos(delta))**2)
    return dxdt


def midpoint_rule(fun,tspan, y0, number_of_points=1000):
    '''Applies the midpoint rule on the system of 4 differential equations for task 10. \n
    fun = function 
    y0 = vector of initial conditions
    optional:\n
    number_of_points = how many steps. Default set to 100. Increasing this reduces error but increases computation time. '''
    dt = (tspan[1] - tspan[0])/number_of_points
    t = np.linspace(tspan[0], tspan[1], number_of_points+1)
    y = np.zeros((number_of_points+1, 4))
    y[0,0] = y0[0] ## Setting up the initial conditions
    y[0,1] = y0[1]
    y[0,2] = y0[2]
    y[0,3] = y0[3]
    for i in range(number_of_points):
        k1 = fun(t[i], y[i,:])
        k2 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k1)
        y[i+1,:] = y[i,:] + dt * k2
    return t, y

#(theta1, omega1, theta2, omega2)
ini_cond_vec = [7*np.pi/180, 1, 0.9*np.pi, 1] # Initial conditions a vector
tspan = [0,10]

approx_sol = midpoint_rule(der_system_task10, tspan, ini_cond_vec) #Midpoint used because its better than euler and easy to implement :)










# from:
# http://www.physics.umd.edu/hep/drew/pendulum2.html
# We can get the equations for the kinetic and potential energy