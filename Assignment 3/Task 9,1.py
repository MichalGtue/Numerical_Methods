#Taking the more or less the same code from task 1,2,3,5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate

#Table print out (same as task 5.1)

def third_order_derivative(t, Ca, k=0.01): # I know that the assignemnt does not require us to add k as an argument but I wanted the function to be as universal as possible
    '''Returns the concentration over time for a third order chemical reaction. 
    t = time in secons
    Ca = concentration of component a (a -> b)
    k = reaction rate constant (default set to 0.01 as this is the value for task 1.)'''
    dca_dt = - k * Ca**3
    return dca_dt

def third_order_analytical(t, c0, k3):
    '''Returns the concentration at a given time for a third order irreversible chemical reation.
    t = Time in seconds. If a vector it will retrun a vector of solutions
    c0 = initial concentration.
    k3 = reaction rate constant for the third order reaction.'''
    Ca = (1/(c0**(-2) + 2*k3*t))**0.5 ## Analytical solution
    return Ca

## We need to slightly modify it so that it only solves for one method

## We see a simplified version of the master function because we only solve one ODE at a time not a system
def midpoint_rule(fun,tspan, y0, number_of_points=100):
    '''Applies the midpoint rule on the system of two differential equations for task 7. \n
    fun = function 
    y0 = vector of initial conditions
    optional:\n
    number_of_points = how many steps. Default set to 100. Increasing this reduces error but increases computation time. '''
    dt = (tspan[1] - tspan[0])/number_of_points
    t = np.linspace(tspan[0], tspan[1], number_of_points+1)
    y = np.zeros(number_of_points+1)
    y[0] = y0 ## Setting up the initial conditions
    for i in range(number_of_points):
        k1 = fun(t[i], y[i])
        k2 = fun(t[i] + dt*0.5, y[i] + 0.5*dt*k1)
        y[i+1] = y[i] + dt * k2
    return t, y


initial_conc = 5
t_max = 100 # For tmax = 100 number of points must be 100 000 to have a dt of 0.001
x_plotting = np.linspace(0, t_max, 1000)
y_analytical = third_order_analytical(x_plotting, initial_conc, 0.01)

N = np.linspace(100, 100000, 10, dtype=int)
solution_list = []
error_list = []
#all comparing at t=4
compare_time = 4
rate_of_convergence = ['N/A']
for i in range(len(N)):
    approx_sol_general = midpoint_rule(third_order_derivative, [0, 4], initial_conc, N[i])
    approx_sol = approx_sol_general[1][-1]  # Gets the last y value corresponding to t = 4
    exact_sol = third_order_analytical(4, initial_conc, 0.01)
    solution_list.append(approx_sol)
    rel_error = np.abs((approx_sol - exact_sol)/exact_sol)
    error_list.append(rel_error)
    if i > 0:  # Doesn't make sense to calculate for the first point
        r = (np.log(error_list[i]/error_list[i-1]))/(np.log(N[i-1] / N[i]))
        rate_of_convergence.append(r)

rows = []
for i in range(len(N)):
    rows.append([N[i], solution_list[i], error_list[i], rate_of_convergence[i]])

df = pd.DataFrame(rows, columns = ['N_t', 'Calculated solution', 'epsilon_rel', 'Rate of Convergence'])
print(df)

# Rate approaches 2. This behaviour is expected.