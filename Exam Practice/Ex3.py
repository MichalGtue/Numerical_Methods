import numpy as np
import scipy.optimize 


def func(x):
    n  = x.size
    fnc = np.zeros(n)
    fnc[0] = 3*x[0] -np.cos(x[2]*x[1]) - 1/2
    fnc[1] = x[0]**2 - x[1] - x[2]
    fnc[2] = x[0] + 1/2*x[1] - np.exp(x[2])
    return fnc

print(scipy.optimize.fsolve(func, x0=[0,1,1]))

