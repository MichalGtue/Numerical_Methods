import scipy.integrate
import matplotlib.pyplot as plt ## For plotting 
import numpy as np

def third_order_derivative(t, Ca, k3=0.01): # I know that the assignemnt does not require us to add k as an argument but I wanted the function to be as universal as possible
    '''Returns the concentration over time for a third order chemical reaction. 
    t = time in secons
    Ca = concentration of component a (a -> b)
    k = reaction rate constant (default set to 0.01 as this is the value for task 1.)'''
    dca_dt = - k3 * Ca**3
    return dca_dt

def third_order_analytical(t, c0, k3):
    '''Returns the concentration at a given time for a third order irreversible chemical reation.
    t = Time in seconds. If a vector it will retrun a vector of solutions
    c0 = initial concentration.
    k3 = reaction rate constant for the third order reaction.'''
    Ca = (1/(c0**(-2) + 2*k3*t))**0.5 ## Analytical solution
    return Ca

initial_conc = 5
t_max = 100
x_plotting = np.linspace(0, t_max, 1000)
y_analytical = third_order_analytical(x_plotting, initial_conc, 0.01)
y_scipy = scipy.integrate.solve_ivp(lambda t, Ca: third_order_derivative(t, Ca), [0, t_max], [initial_conc])
y_scipy_plotting = y_scipy.y[0]
x_scipy_plotting = y_scipy.t
fig = plt.figure()
plt.plot(x_plotting, y_analytical, 'r', label="Analytical Solution")
plt.plot(x_scipy_plotting, y_scipy_plotting, 'b', label='Scipy solve_ivp')
plt.xlabel('Time (s)')
plt.ylabel('Concentration of A (mol/L)')
plt.title(f'Concentration of A over time with initial concentration of {initial_conc} mol/L', weight='bold')
plt.legend()
plt.show()
## The figure above shows that they are almost identical


## We can restrict the domain to see the deviations
fig2 = plt.figure()
plt.plot(x_plotting, y_analytical, 'r', label="Analytical Solution")
plt.plot(x_scipy_plotting, y_scipy_plotting, 'b', label='Scipy solve_ivp')
plt.xlabel('Time (s)')
plt.ylabel('Concentration of A (mol/L)')
plt.title(f'Closer look at the two solution methods', weight='bold')
plt.xlim(35, 61)
plt.ylim(0.85, 1.2)
plt.legend()
plt.show()
#Slight deviations but we need to zoom in quite close to see

# For difference over time 
y_analytical_scipyxvalues = third_order_analytical(x_scipy_plotting, initial_conc, 0.01)

diff = []

for i in range(len(x_scipy_plotting)):
    difference = y_analytical_scipyxvalues[i] -  y_scipy_plotting[i]
    diff.append(difference)

fig3 = plt.figure(figsize=(8,7))
plt.plot(x_scipy_plotting, diff)
plt.xlabel('Time(s)')
plt.ylabel('Analytical - solve_ivp')
plt.title('Difference between analytical and numerical solution', weight='bold')
plt.show()