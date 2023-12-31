import numpy as np
import matplotlib.pyplot as plt

# This file is similar to 'task5.2'
# Here we try to find the minimum number of iterations (max dt) to yield a stable result

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

# Same as task 3
initial_conc = 5
t_max = 100
x_plotting = np.linspace(0, t_max, 1000)
y_analytical = third_order_analytical(x_plotting, initial_conc, 0.01)
solution_midpoint_13 = midpoint_rule(third_order_derivative, [0, t_max], initial_conc, 13) # dt ~= 7.69
solution_midpoint_14 = midpoint_rule(third_order_derivative, [0, t_max], initial_conc, 14) # dt ~= 7.14
solution_midpoint_15 = midpoint_rule(third_order_derivative, [0, t_max], initial_conc, 15) # dt ~= 6.67
solution_midpoint_20 = midpoint_rule(third_order_derivative, [0, t_max], initial_conc, 20) # dt ~= 5
solution_midpoint_50 = midpoint_rule(third_order_derivative, [0, t_max], initial_conc, 50)
solution_midpoint_100 = midpoint_rule(third_order_derivative, [0, t_max], initial_conc, 100)


fig = plt.figure()
plt.plot(x_plotting, y_analytical, 'r', label="Analytical Solution")
plt.plot(solution_midpoint_13[0], solution_midpoint_13[1], label='Midpoint Rule N=13') # [0] holds the x position and [1] holds the 'y' position
plt.plot(solution_midpoint_14[0], solution_midpoint_14[1], 'b', label='Midpoint Rule N=14') # [0] holds the x position and [1] holds the 'y' position
plt.plot(solution_midpoint_15[0], solution_midpoint_15[1], 'g', label='Midpoint Rule N=15') # Though it isnt perfect, at least now it doesnt go down to minus infinity.
plt.plot(solution_midpoint_20[0], solution_midpoint_20[1], 'orange', label='Midpoint Rule N=20') 
plt.plot(solution_midpoint_50[0], solution_midpoint_50[1], 'purple', label='Midpoint Rule N=50') 
plt.plot(solution_midpoint_100[0], solution_midpoint_100[1], label='Midpoint Rule N=100') 
# N=13 is a straight line => Not expected and cannot be used.
# N=14 is getting there but still not perfect. May be used for large values but not for small
# N=15 is the first function that starts converging at a reasonable time.
# N=20 begins to approximate it better and then it can be seen that increasing the number of iterations will improve the approximation.
# N=100 nearly overlaps the solution.
plt.xlabel('Time (s)')
plt.ylabel('Concentration of A (mol/L)')
plt.title(f'Concentration of A over time with initial concentration of {initial_conc} mol/L', weight='bold')
plt.legend()
plt.ylim(-0.1, 5.5)
plt.show()

