import numpy as np
import matplotlib.pyplot as plt


def der(x,y):
    k=1
    return -k*y 

def euler_method(func,x,y, end, dt=0.1):
    k = 1
    x_i = x
    y_i = y
    while x_i < end:
        x_i1 = x_i + dt
        y_i1 = y_i + dt * func(k,y_i)
        #print(x_i1, y_i1)
        x_i = x_i1
        y_i = y_i1

    return y_i

print(euler_method(der, 0, 1, 2))


def euler_imp(func,x,y, end, dt=0.1):
    x_i = x
    y_i = y
    while x_i < end:
        x_i1 = x_i + dt 
        y_i1 = y_i + (dt * (-1*y_i))/(1-dt * func(x_i,y_i))
        y_i = y_i1
        x_i = x_i1
        #print(x_i1,y_i1)
    return x_i1, y_i1

print(euler_imp(der, 0, 1, 2))



#Class solution
def func(t, c):
    k=1
    dcdt = -k*c
    return dcdt

def euler_class(f, tspan,c0, n=100):
    dt = (tspan[1] - tspan[0])/n
    t = np.linspace(tspan[0], tspan[1], n+1)
    c = np.zeros(n+1)
    c[0] = c0
    for i in range(n):
        k1 = f(t[i],c[i])
        c[i+1] = c[i] + dt*k1
    return t,c

def euler_class_imp(f, tspan,c0, n=100):
    h = 1e-8
    dt = (tspan[1] - tspan[0])/n
    t = np.linspace(tspan[0], tspan[1], n+1)
    c = np.zeros(n+1)
    c[0] = c0
    for i in range(n):
        fff = f(t[i], c[i])
        ffh = f(t[i], c[i]+h)
        dfdc = (ffh-fff)/h
        c[i+1] = c[i] + dt*fff/(1-0.5*dt*dfdc)
    return t,c

k=1
t, c_expl = euler_class(lambda t,c: -k*c, [0,2], 1, 20)
t, c_imp = euler_class_imp(lambda t,c: -k*c, [0,2], 1, 20)

fig, ax = plt.subplots()

ax.plot(t, c_expl)
ax.plot(t, c_imp)

plt.show()