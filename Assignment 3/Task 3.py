import scipy.integrate
import matplotlib.pyplot as plt ## For plotting 
import numpy as np

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

