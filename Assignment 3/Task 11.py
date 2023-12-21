import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import pandas as pd
import time
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

#(theta1, omega1, theta2, omega2)
ini_cond_vec = [7*np.pi/180, 1, 0.9*np.pi, 1] # Initial conditions a vector
tspan = [0,10]

approx_sol = master_function(der_system_task10, tspan, ini_cond_vec, 'midpoint', 2000) #Midpoint used because its better than euler and easy to implement :)


# from:
# http://www.physics.umd.edu/hep/drew/pendulum2.html
# We can get the equations for the kinetic and potential energy
# the time derivative in terms of theta1 is simply omega1

KE = 0.5 * (m1_value+m2_value) * l1_value**2 * approx_sol[1][:,1]**2 + 0.5 * m2_value* l2_value**2 * approx_sol[1][:,3]**2 + m2_value*l1_value*l2_value * approx_sol[1][:,1] * approx_sol[1][:,3] * np.cos(approx_sol[1][:,0] - approx_sol[1][:,2])

PE = -(m1_value+m2_value) * 9.81 *l1_value *np.cos(approx_sol[1][:,0]) - m2_value*9.81*l2_value*np.cos(approx_sol[1][:,2])


## Correcting for the zero of energy
PE += (m1_value+m2_value)*9.81*(l1_value+l2_value)
## Without this correction we would see that the potential energy would be negative and that the sum of energies would be negative. 
## Here we set the zero of energy to the lowest possible point the pendulum can reach.
## With l1=1 and l2=1 the lowest point would be y=-2. Thus, we must correct for this.



fig = plt.figure()
plt.plot(approx_sol[0], KE, label='Kinetic Energy')
plt.plot(approx_sol[0], PE, label='Potential Energy')
plt.plot(approx_sol[0], KE + PE, label='Sum of both')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title(f'theta1={ini_cond_vec[0]: .4f}, omega1={ini_cond_vec[1]}, theta2={ini_cond_vec[2]: .4f}, omega2={ini_cond_vec[3]}', fontsize=10)
plt.suptitle('Potential and Kinetic Energy of the Double Pendulum')
plt.legend()
plt.show()
## We can see that the graph matches expectations and the total energy is conserved.
## Moreover, the shape of the graphs alligns with the expectations that as kinetic energy increases potential decreases and vice versa.+


methods = ['midpoint', 'euler', 'rk2', 'rk4']
rows = []

for i in range(len(methods)):
    start_time = time.time()
    solution_loop = master_function(der_system_task10, tspan, ini_cond_vec, 'midpoint', 2000)
    end_time = time.time() - start_time
    KE_loop = 0.5 * (m1_value+m2_value) * l1_value**2 * solution_loop[1][:,1]**2 + 0.5 * m2_value* l2_value**2 * solution_loop[1][:,3]**2 + m2_value*l1_value*l2_value * solution_loop[1][:,1] * solution_loop[1][:,3] * np.cos(solution_loop[1][:,0] - solution_loop[1][:,2])
    PE_loop = -(m1_value+m2_value) * 9.81 *l1_value *np.cos(solution_loop[1][:,0]) - m2_value*9.81*l2_value*np.cos(solution_loop[1][:,2])
    PE_loop += (m1_value+m2_value)*9.81*(l1_value+l2_value)
    rows.append([methods[i], np.median(KE_loop+PE_loop), end_time])

df = pd.DataFrame(rows, columns = ['Method Used', 'Total Energy', 'Time taken (s)'])
print(df)
