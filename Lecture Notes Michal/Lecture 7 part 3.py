import numpy as np
from scipy.optimize import fsolve

def eqns(x):
    f = np.zeros_like(x)
    f[0] = x[0]**2 + x[1]**2 - 4
    f[1] = x[0]**2 - x[1] +1
    return f

sol = fsolve(eqns, [1.0,2.0], full_output=True)
print(sol)