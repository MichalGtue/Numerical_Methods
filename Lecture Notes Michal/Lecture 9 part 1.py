## Eulers method
import numpy as np
import matplotlib.pyplot as plt

def der(k,c):
    return -k*c 

def euler_method(func,x,y, end, dt=0.1):
    k = 1
    x_i = x
    y_i = y
    while x_i < end:
        x_i1 = x_i + dt
        y_i1 = y_i + dt * func(k,y_i)
        print(x_i1, y_i1)
        x_i = x_i1
        y_i = y_i1

    return y_i

print(euler_method(der, 0, 1, 2))


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

t, c =euler_class(lambda t,c: -200*c, [0,2], 1, 20)

print(np.vstack([t,c]).T)


plt.plot(t,c, marker='.')
plt.show()