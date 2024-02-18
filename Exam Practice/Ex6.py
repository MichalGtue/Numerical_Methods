import scipy.integrate
import numpy as np


def func(x):
    return (np.sin(np.pi*x/2))/(x) + 1

print(scipy.integrate.quad(func, 0,10))