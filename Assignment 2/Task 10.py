#Libraries
import numpy as np
import sympy
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
import scipy.integrate
import pandas as pd

#Prerequisites from the previous task
a = sympy.Symbol('x', real=True) #x is already used for the list
p = sympy.Symbol('y', real=True) #y already used for list as well
#From assignment document
x = list ( range (1 ,12 ,2) )
y = [13.40 ,15.00 ,22.0 ,16.70 ,10.60 ,8.30]

def Lagrange_interpolation(g,h):
    '''Returns the Lagrange polynomial using Lagrange interpolation.
    Arguments g and h both of which must be lists'''
    assert len(g) == len(h), "Every x coordinate must have a corresponding y coordinate"
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
y_for_plotting_lagrange = function(x_for_plotting)
spline_eq = scipy.interpolate.splrep(x,y)
y_for_plotting_spline = scipy.interpolate.splev(x_for_plotting, spline_eq)
def f(t):
    return -0.1 * t**3 + 0.58 * t**2 * np.cosh((-1)/(t+1)) + np.exp(t/2.7) + 9.6


## Below, we can see the integral of the interpolated function and that of q 
print(f'Integral of the spline interpolant: {scipy.integrate.quad(lambda x: scipy.interpolate.splev(x, spline_eq), 0, 12)[0]}')
print(f'Integral of the Lagrange interpolant: {scipy.integrate.quad(function, 0, 12)[0]}')
print(f'Integral of the function from task 6: {scipy.integrate.quad(lambda x: f(x), 0, 12)[0]}')

## We can see that the integral from task 6 seems to underestimate while the initial assumption seems to overestimate. The two interpolants lie in between. 
## What we can do is assume that our real source for lake superior lies somewhere in between 160 and 180
## Finally we can run a simulation at varrying inlet concentrations of lake superior and see the results,
## on the lakes that are dependent on it (All except for Lake Michigan)

## Below is code from task 5 required to perform the calculations

#task 2 matrix
#Define the values of the Q's (given in km^-3 / yr)
Q_SH = 72
Q_MH = 38
Q_HE = 160
Q_EO = 185
Q_OO = 215

#Define values for the different PCB source S_in's (given in kg/yr)

# Now S_in can take different values
S_in_M = 810
S_in_H = 630
S_in_E = 2750
S_in_O = 3820

#Volumes (km^3)

V_S = 12000
V_M = 4900
V_H = 3500
V_E = 480
V_O = 1640

#Defining k1
k1 = 0.00584

#Defining k2
k2 = 2 * 10**7

#Volume of tank is given in m^3 but everything else is in km^3. Thus, the volume is given in km^3
V_tank = 1*10**-9


# Not given in the question so taken as 100
Q_MO = 20



## For PFR Graph
# Making the matrix

PFR_matrix = np.zeros((1005,1005))
new_sol_vec = np.zeros(1005)

M = np.array([[Q_SH + k1*V_S,0,0,0,0], [0,Q_MH + k1*V_M + Q_MO,0,0,0], [-Q_SH, -Q_MH, Q_HE + k1*V_H, 0, 0], [0,0,-Q_HE, Q_EO + k1*V_E, 0], [0, 0,0,-Q_EO,Q_OO + k1*V_E]])
PFR_matrix[:5, :5] = M # Update the new matrix 
PFR_matrix[4, -1] = -Q_MO # Inflow from the last PFR
PFR_matrix[5, 1] = Q_MO # First PFR is dependent on the concentration of lake Michigan
PFR_matrix[5, 5] = -(Q_MO + V_tank*k2)

for i in range(999):
    PFR_matrix[6+i, 5+i] = Q_MO 
    PFR_matrix[6+i, 6+i] = -(Q_MO + V_tank*k2)


x_task_10 = []
Superior_list = []
Huron_list = []
Erie_list = []
Ontario_list = []
for j in range(81): # 
    x_task_10.append(160 + j/4) # Will be needed later to generate our plots
    sol_vec = np.array([160 + j/4, S_in_M, S_in_H,S_in_E, S_in_O])
    new_sol_vec[:5] = sol_vec
    sup, mich, hur, er, ont = np.linalg.solve(PFR_matrix, new_sol_vec)[0:5] # solve the matrix and store the solutions
    Superior_list.append(sup) ## Appends the solutions to the lists
    Huron_list.append(hur) ## Needed for plotting
    Erie_list.append(er)
    Ontario_list.append(ont)


fig = plt.figure(figsize=(12,10))
fig.suptitle('PCB concentration in various lakes at different inlet concentration to Lake Superior', size=10, weight='bold')

axsup = plt.subplot(2,2,1)
axsup.plot(x_task_10, Superior_list)

axhur = plt.subplot(2,2,2)
axhur.plot(x_task_10, Huron_list)

axer = plt.subplot(2,2,3)
axer.plot(x_task_10, Erie_list)

axont = plt.subplot(2,2,4)
axont.plot(x_task_10, Ontario_list)

#y labels
axont.set_ylabel('PCB concentration (kg/km^3)')
axer.set_ylabel('PCB concentration (kg/km^3)')
axhur.set_ylabel('PCB concentration (kg/km^3)')
axsup.set_ylabel('PCB concentration (kg/km^3)')

#x labels
axont.set_xlabel('PCB source of Lake Superior (kg/year)')
axer.set_xlabel('PCB source of Lake Superior (kg/year)')
axhur.set_xlabel('PCB source of Lake Superior (kg/year)')
axsup.set_xlabel('PCB source of Lake Superior (kg/year)')

#axis titles
axont.set_title('Lake Ontario')
axer.set_title('Lake Erie')
axhur.set_title('Lake Huron')
axsup.set_title('Lake Superior')


plt.show()


def derivative(q,w):
    '''Returns the derivative of a linear function. First argument is the x list and second arument is the y list'''
    return (w[-1]-w[0])/(q[-1]-q[0])

rows = [['Lake Superior', derivative(x_task_10, Superior_list)], ['Lake Huron', derivative(x_task_10, Huron_list)], ['Lake Erie', derivative(x_task_10, Erie_list)], ['Lake Ontario', derivative(x_task_10, Ontario_list)]]

df = pd.DataFrame(rows, columns = ['Lake', 'Rate of change'])

print(df)