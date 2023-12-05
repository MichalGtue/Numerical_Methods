#Libraries
import numpy as np
import sympy
import matplotlib.pyplot as plt

#Prerequisites from the previous task, needed to plot the lagrange interpolant
a = sympy.Symbol('x', real=True) #x is already used for the list
p = sympy.Symbol('y', real=True) #y already used for list as well
#From assignment document
x = list ( range (1 ,12 ,2) )
y = [13.40 ,15.00 ,22.0 ,16.70 ,10.60 ,8.30]

def Lagrange_interpolation(g,h):
    '''Returns the Lagrange polynomial using Lagrange interpolation.
    Arguments g and h both of which must be lists'''
    assert len(g) == len(h), "Every g coordinate must have a corresponding h coordinate"
    output_storage = 0
    for k in range(len(g)):
        polynomial_storage = 1  ## Defining polynomial storage and resetting after every k
        for i in range(len(g)):
            if i!=k:
                polynomial_storage = polynomial_storage * (a-g[i])/(g[k]-g[i])
        output_storage = output_storage + h[k]*polynomial_storage
    output_storage = sympy.simplify(output_storage) #This line is not exactly necessary but the function become really ugly otherwise
    return output_storage
#Needed to generate the plots of the lagrange interpolant
function = sympy.lambdify(a, Lagrange_interpolation(x,y))
x_for_plotting = np.linspace(0, 12, 1000)
y_for_plotting = function(x_for_plotting)


#The actual new code starts here










fig = plt.subplot()
ax1 = fig.plot(x, y, marker='o', linestyle='None')
ax2 = fig.plot(x_for_plotting, y_for_plotting)

plt.show()