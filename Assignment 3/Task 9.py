#Taking the more or less the same code from task 1,2,3,5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate


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


y_scipy = scipy.integrate.solve_ivp(lambda t, Ca: third_order_derivative(t, Ca), [0, t_max], [initial_conc])
y_scipy_plotting = y_scipy.y[0]
x_scipy_plotting = y_scipy.t