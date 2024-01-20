import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def func(x):
    n = x.size
    fnc = np.zeros(n)
    fnc[0] = 3*x[0] - np.cos(x[1]*x[2]) - 1/2
    fnc[1] = x[0]**2 - x[1] - x[2]
    fnc[2] = x[0] + 0.5*x[1] - np.exp(x[2])
    return fnc


def jac_num(func, x):
    n = x.size
    jac = np.eye(n)
    h = 1e-8
    f = func(x)
    for i in range(n):
        xs = x[i]
        x[i] = x[i] + h
        fh = func(x)
        jac[:,i] = (fh - f)/h
        x[i] = xs
    return jac




def newton(func, jac, x, tol_x, tol_f):
    itmax = 100
    converged = False
    it = 0
    x = np.asarray(x)
    f = func(x)
    while (not converged) and (it<itmax):
        it += 1
        j = jac_num(func, x)
        dx = np.linalg.solve(j,-f)
        x = x + dx
        f = func(x)
        converged = (np.max(abs(dx))<=tol_x) and (np.max(abs(f))<=tol_f)
    return x

def broyden(func, x, tol_x, tol_f):
    itmax = 100
    converged = False
    it = 0
    x = np.asarray(x, dtype = float)
    f = func(x)
    #j = np.eye(len(x))
    j = jac_num(func, x)
    while (not converged) and (it<itmax):
        it += 1
        dx = np.linalg.solve(j,-f)
        x = x + dx
        f0 = np.copy(f)
        f = func(x)
        df = f - f0
        j = j + np.outer(df - np.dot(j,dx), dx)/np.dot(dx, dx)
        converged = (np.max(abs(dx))<=tol_x) and (np.max(abs(f))<=tol_f)
    return x
sol = broyden(func, [1, 2,1], 1e-12, 1e-12)


print(sol)


print(scipy.optimize.fsolve(func, x0=[1,2,1]))