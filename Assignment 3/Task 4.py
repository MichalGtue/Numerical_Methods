import numpy as np
import matplotlib.pyplot as plt
def first_order_euler(fun, tspan, y0, number_of_points=100000): # 100_000 corresponds to to a dt of 0.001 as asked in the question
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

#From task 1
# I know I could use something like "from Task 1 import third_order_derivative" but I want to make sure that the code runs on all devices always.
# For some people importing from other files doesn't work.

def third_order_derivative(t, Ca, k=0.01): 
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
t_max = 100 # For tmax = 100, the number of points must be 100 000 to have a dt of 0.001
x_plotting = np.linspace(0, t_max, 1000)
y_analytical = third_order_analytical(x_plotting, initial_conc, 0.01)
solution_euler = first_order_euler(third_order_derivative, [0, t_max], initial_conc)



fig = plt.figure()
plt.plot(x_plotting, y_analytical, 'r', label="Analytical Solution")
plt.plot(solution_euler[0], solution_euler[1], 'b', label='First Order Euler') # [0] holds the x position and [1] holds the 'y' position
plt.xlabel('Time (s)')
plt.ylabel('Concentration of A (mol/L)')
plt.title(f'Concentration of A over time with initial concentration of {initial_conc} mol/L', weight='bold')
plt.legend()
plt.ylim(-0.1, 5.5)
plt.show()


plt.plot(x_plotting, y_analytical, 'r', label="Analytical Solution")
plt.plot(solution_euler[0], solution_euler[1], 'b', label='First Order Euler') # [0] holds the x position and [1] holds the 'y' position
plt.xlabel('Time (s)')
plt.ylabel('Concentration of A (mol/L)')
plt.title(f'Closer Look at the methods to observe slight deviations', weight='bold')
plt.legend()
plt.ylim(1.896, 1.897)
plt.xlim(11.8950, 11.9125)
plt.show()