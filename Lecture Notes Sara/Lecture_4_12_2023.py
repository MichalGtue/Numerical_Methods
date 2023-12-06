import numpy as np
def newton(func, grad, x, tol_x, tol_f):
    itmax=100
    h=1e-8
    converged=False
    it=0
    f=func(x)
    while (not converged) and (it<itmax):
        it += 1
        #g=grad(x)
        g=(func(x+h)-f)/h
        dx=-f/g
        x+= dx
        f=func(x)
        converged = (abs(dx)<=tol_x) and (abs(f)<=tol_f)
        print(f"it:{it:2}, x:{x:18.16f}, f(x):{f:15.8e}")
    if converged:
        print(f"Newton: Toot not found {it} iterations at x={x:23.15e} with function value {f:15.8e}")
    else:
        print(f"Newton error: maximum number iterations possible")


    return x

#the code does not work with x=2 as gradient =0

newton(lambda x: x**2-4*x+2, lambda x: 2*x-4, 1, 1e-15, 1e-15)
#use brands method for 1D as faster because avoids diverging and

import numpy as np
def func(x):
    n=x.size
    fnc=np.zeros(n)
    fnc[0]=x[0]**2+x[1]**2-4
    fnc[1]=x[0]**2+x[1]**2+1
    return fnc
def jac(x):
    n=x.size
    jac=np.zeros((n,n))
    jac[0,0]=2*x[0]
    jac[0,1]=2*x[1]
    jac[1,0]=2*x[0]
    jac[1,1]=-1
    return jac

import numpy as np
def newton(func, grad, jac, x, tol_x, tol_f):
    itmax=100
    #h=1e-8
    j=jac(x)
    converged=False
    it=0
    f=func(x)
    while (not converged) and (it<itmax):
        it += 1
        #g=grad(x)
        g=(func(x+h)-f)/h
        #dx=-f/g
        dx=np.linalg.solve(j,-f)
        x+= dx
        f=func(x)
        #converged = (abs(dx)<=tol_x) and (abs(f)<=tol_f)
        converged = (np.max(abs(dx))<=tol_x) and (np.max(abs(f))<=tol_f)
        #print(f"it:{it:2}, x:{x:18.16f}, f(x):{f:15.8e}")
        print(f"it:{it:2}, x[0]: {x[0]:18.16f}, x[1]: {x[1]:18.16f}, f0:{f[0]:15.8e}, f1:{f[1]:15.8e}")
    if converged:
        print(f"Newton: Toot not found {it} iterations at x={x:23.15e} with function value {f:15.8e}")
    else:
        print(f"Newton error: maximum number iterations possible")


    return x

#the code does not work with x=2 as gradient =0

newton(lambda x: x**2-4*x+2, lambda x: 2*x-4, 1, 1e-15, 1e-15)
#use brands method for 1D as faster because avoids diverging and 
#jacobi expensive with big atrix, therefore use Broyden's method for big matrix

import numpy as np
from scipy.optimize import root_scalar

coeff = {'a':1, 'b':-4, 'c':2}
def func(x, coeff):
    return coeff['a']*x**2+coeff['b']*x+coeff['c']

#sol=root_scalar(lamda x: x**2-4*x+2, bracket=[-1,1], method='brenth')
#sol=root_scalar(lamda x: x**2-4*x+2, bracket=[-1,1], method='bisect')
# option 1 sol=root_scalar(func, args=coeff, bracket=[-1,1], method='brenth') but slower than below eq, use below
sol=root_scalar(lamda x: func(x, coeff), bracket=[-1,1], method='brenth')
print(sol)



import numpy as np
from scipy.optimize import root_scalar

param={'conv':0.99, 'k':1}
def func(tau, param):
    return param['conv']-(1-np.exp(-param['k']*tau))


sol=root_scalar(lamda tau: func(tau, param), bracket=[0,10], method='brenth')
print(sol)


import numpy as np
from scipy.optimize import fsolve

def eqns(x):
    f=np.zeros_like(x)
    f[0]=x[0]**2+x[1]**2-4
    f[1]=x[0]**2-x[1]+1
    return f 

sol = fsolve(eqns, [1.0,2.0], full_output=True)
print(sol)
