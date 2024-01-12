import numpy as np
import matplotlib.pyplot as plt


#Class solution
def func(t, c):
    k=1
    dcdt = -k*c
    return dcdt

def euler_class_vec(f, tspan,c0, n=100):
    dt = (tspan[1] - tspan[0])/n
    t = np.linspace(tspan[0], tspan[1], n+1)
    c0 = np.asarray(c0, dtype=float)
    c= np.zeros((n+1, c0.size))
    c[0] = c0
    print(f't: {t[0]:f}, c: {np.array2string(c[0])}')
    for i in range(n):
        k1 = f(t[i],c[i])
        c[i+1] = c[i] + dt*k1
        print(f't: {t[0]:f}, c: {np.array2string(c[i+1])}')
    return t,c

def jac(func, t, c):
    n = c.size
    jac = np.zeros((n,n))
    h = 1e-8
    f = func(t, c)
    for i in range(n):
        cs = c[i]
        c[i] = c[i] + h
        fh = func(t,c)
        jac[:,i] = (fh-f)/h
        c[i] = cs
    return jac



def euler_class_imp(func, tspan,c0, n=100):
    dt = (tspan[1] - tspan[0])/n
    t = np.linspace(tspan[0], tspan[1], n+1)
    c0 = np.asarray(c0, dtype=float)
    c= np.zeros((n+1, c0.size))
    c[0] = c0
    iden = np.eye(c0.size)
    print(f't: {t[0]:f}, c: {np.array2string(c[0])}')
    for i in range(n):
        f = func(t[i], c[i])
        dfdc = jac(func, t[i], c[i])
        dc = np.linalg.solve(iden-0.5*dt*dfdc, dt*f)
        c[i+1] = c[i] + dc
        print(f't: {t[0]:f}, c: {np.array2string(c[i+1])}')
    return t,c

def func2(t,c):
    k1=10.0
    k2=1.0
    r1=k1*c[0]
    r2 = k2*c[1]
    f = np.zeros(c.size)
    f[0] = -r1
    f[1] = r1-r2
    f[2] = r2
    return f


t, c =euler_class_imp(func2, [0,2], [1,0,0], 100)

c= c.T

fig = plt.figure()

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.plot(t,c[0], 'ro-', label='A')
plt.plot(t,c[1], 'bs-', label='B')
plt.plot(t,c[2], 'g^-', label='C')

plt.legend(loc='upper center')
plt.grid()
plt.show()