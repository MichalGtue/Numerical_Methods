import numpy as np
import scipy as sp
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt

def tu_ode_fun(t,u, k1, k2):
    dudt = -k1*u +k2
    return dudt

def model_results(t, k1, k2):
    tspan = [min(t), max(t)]
    init = [1.0]
    sol = sp.integrate.solve_ivp(tu_ode_fun, tspan, init, args = (k1, k2), t_eval = t)
    return sol.y[0]


T, U = np.loadtxt('Lecture Notes Michal/tudataset1.txt', unpack = True, skiprows = 1)
tspan = [min(T), max(T)]
init = [U[0]]
k1 = 2.0
k2 = 0.5


coeffs = sp.optimize.curve_fit(model_results, T,U, p0 = (k1, k2))[0]
k1, k2 = coeffs
sol = sp.integrate.solve_ivp(tu_ode_fun, tspan, init, args = (k1,k2), t_eval=T)

plt.plot(T,U, 'x', label = 'Experiment')
plt.plot(sol.t,sol.y[0], label = 'Simulation')
plt.show()

