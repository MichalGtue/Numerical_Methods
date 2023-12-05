import numpy as np
from scipy.optimize import root_scalar



coeff = {'a':1, 'b':-4, 'c':2}
def func(x, coeff):
    return  coeff['a']*x**2 + coeff['b']*x + coeff['c']


param = {'conv': 0.99, 'k':1}

def func2(tau, param):
    return param['conv'] - (1-np.exp(-param['k']*tau))

sol = root_scalar(lambda tau: func2(tau, param), bracket=[0,10], method='brenth')

print(sol)

