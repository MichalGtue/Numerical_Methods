import numpy as np

#Newton method

def fnc(x):
    return x**2-4*x+2

def fncprime(x):
    return 2*x-4


def newton_method(func, funcprime, x1, nguess=100, tol_x=1e-6, tol_f=1e-6):
    converged = False
    counter = 0
    while (not converged) and (counter < nguess):
        x2 = x1 - func(x1)/funcprime(x1)
        x3 = x2
        x1 = x2
        counter +=1
        converged = (abs(x3 - x1) <= tol_x and abs(func(x1) <= tol_f))
        print(counter)
    return x1

def newton_class(func,x,tol_x,tol_f):
    '''Newton-raphson, no input of derivative info'''
    itmax = 100
    converged = False
    h=1e-8
    it = 0
    f = func(x)
    while(not converged) and (it<itmax):
        it +=1
        g = (func(x+h) - f)/h
        dx = -f/g
        x +=dx
        f = func(x)
        converged = (abs(dx) <= tol_x) and (abs(f) <= tol_f)
        print(it)
    if converged:
        print(f" Newton root found in {it}")
    return x

#print(newton_class(fnc, 5, 1e-15, 1e-15))


# Newton raphson for multidimensional



def func(x):
    n  = x.size
    fnc = np.zeros(n)
    fnc[0] = x[0]**2 + x[1]**2 - 4
    fnc[1] = x[0]**2 - x[1] + 1
    return fnc

def jac(x):
    n = x.size
    jac = np.zeros((n,n))
    jac[0,0] = 2*x[0]
    jac[0,1] = 2*x[1]
    jac[1,0] = 2*x[0]
    jac[0,1] = -1
    return jac


def newton_class_multivariable(func,x,tol_x,tol_f):
    '''Newton-raphson, no input of derivative info. Multivariable'''
    itmax = 100
    converged = False
    x = np.asarray(x)
    it = 0
    f = func(x)
    while(not converged) and (it<itmax):
        it +=1
        g = jac(x)
        dx = np.linalg.solve(g, -f)
        x = x + dx
        f = func(x)
        converged = (np.max(abs(dx)) <= tol_x) and (np.max(abs(f)) <= tol_f)
        print(f"it:{it:2}, x[0]: {x[0]:18.16f}, x[1]: {x[1]:18.16f}, f[0]:{f[0]:23.15e}, f[1]:{f[1]:23.15e}")
    if converged:
        print(f" Newton root found in {it}")
    return x

newton_class_multivariable(fnc, [1,2], 1e-16, 1e-16)