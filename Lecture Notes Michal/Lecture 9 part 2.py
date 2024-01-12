import numpy as np
import matplotlib.pyplot as plt
#Class solution
def func(t, c):
    k=1
    dcdt = -k*c
    return dcdt

## Second order method 
def RK2(f, tspan,c0, n=100):
    dt = (tspan[1] - tspan[0])/n
    t = np.linspace(tspan[0], tspan[1], n+1)
    c = np.zeros(n+1)
    c[0] = c0
    for i in range(n):
        k1 = f(t[i],c[i])
        k2 = f(t[i+1], c[i] + dt*k1)
        c[i+1] = c[i] + dt*((k1+k2)/2)
    return t,c

t, c =RK2(lambda t,c: -10*c, [0,2], 1, 20)

print(np.vstack([t,c]).T)


#plt.plot(t,c, marker='.')



def midpoint_rule(f, tspan,c0, n=100):
    dt = (tspan[1] - tspan[0])/n
    t = np.linspace(tspan[0], tspan[1], n+1)
    c = np.zeros(n+1)
    c[0] = c0
    for i in range(n):
        k1 = f(t[i],c[i])
        k2 = f(t[i]+0.5*dt, c[i] + 0.5*dt*k1)
        c[i+1] = c[i] + dt*k2
    return t,c


t1, c1 =midpoint_rule(lambda t,c: -1*c, [0,2], 1, 20)

#plt.plot(t1,c1, marker='.')


def RK4(f, tspan,c0, n=100):
    dt = (tspan[1] - tspan[0])/n
    t = np.linspace(tspan[0], tspan[1], n+1)
    c = np.zeros(n+1)
    c[0] = c0
    for i in range(n):
        k1 = f(t[i],c[i])
        k2 = f(t[i]+0.5*dt, c[i] + 0.5*dt*k1)
        k3 = f(t[i]+0.5*dt, c[i] + 0.5*dt*k2)
        k4 = f(t[i+1], c[i] + dt*k3)
        c[i+1] = c[i] + dt/6 * (k1 + 2*(k2+k3) + k4)
    return t,c

t2, c2 = RK4(lambda t,c: -1*c, [0,2], 1, 20)
plt.plot(t2,c2, marker='.')

plt.show()