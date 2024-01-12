import numpy as np
import matplotlib.pyplot as plt

#Functions from previous tasks
def first_order_euler(fun, tspan, y0, number_of_points=100000):
    '''Explicit first order euler solver for ODEs.\n
    fun = derivative function \n
    tspan = supply the time span as a vector ie [0,10] \n
    y0 = initial condition ie at t=0 y0=100 \n
    number_of_points = Number of points. Default set to 100000 in order to have dt = 0.001. Higher number means longer computation time'''
    dt = (tspan[1] - tspan[0])/number_of_points
    t = np.linspace(tspan[0], tspan[1], number_of_points+1)
    y = np.zeros(number_of_points+1)
    y[0] = y0 ## Setting up the initial conditions
    for i in range(number_of_points):
        y[i+1] = y[i] + dt * fun(t[i], y[i])
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
solution_euler_24 = first_order_euler(third_order_derivative, [0, t_max], initial_conc, 24) # dt ~= 4.1667
solution_euler_25 = first_order_euler(third_order_derivative, [0, t_max], initial_conc, 25) # dt = 4
solution_euler_40 = first_order_euler(third_order_derivative, [0, t_max], initial_conc, 40)
solution_euler_80 = first_order_euler(third_order_derivative, [0, t_max], initial_conc, 80)


fig = plt.figure()
plt.plot(x_plotting, y_analytical, 'r', label="Analytical Solution")
plt.plot(solution_euler_24[0], solution_euler_24[1], 'b', label='First Order Euler N=24') # [0] holds the x position and [1] holds the 'y' position
plt.plot(solution_euler_25[0], solution_euler_25[1], 'g', label='First Order Euler N=25') # Though it isnt perfect, at least now it doesnt go down to minus infinity.
plt.plot(solution_euler_40[0], solution_euler_40[1], 'orange', label='First Order Euler N=40') 
plt.plot(solution_euler_80[0], solution_euler_80[1], 'purple', label='First Order Euler N=80') 

plt.xlabel('Time (s)')
plt.ylabel('Concentration of A (mol/L)')
plt.title(f'Concentration of A over time with initial concentration of {initial_conc} mol/L', weight='bold')
plt.legend()
plt.ylim(-0.1, 5.5)
plt.show()

