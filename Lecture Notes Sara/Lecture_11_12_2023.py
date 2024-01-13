#exercise 1: euler's method
import numpy as np
import math
import matplotlib.pyplot as plt

def func(t,c):
    k=1
    dcdt=-1*c
    return dcdt
#c concentration

def euler(f, t_span, c0, n=100):
    dt=(t_span[1]-t_span[0])/n
    t= np.linspace(t_span[0], t_span[1], n+1)
    c= np.zeros(n+1)
    c[0]= c0
    for i in range(n):
        k1 = f(t[i], c[i])
        t +=dt
        c[i+1] += c[i] + dt*k1
#        print(f"t: {t:1.2f}, c: {c:.6f}")
    return t, c
    # t_spa = t0 to tn range, 

t, c = euler(lambda t,c: -1*c, [0,2], 1, 10)
print(np.vstack([t,c]).T)

plt.plot(t,c, marker= '*')
plt.show()

#Runge-Kutta
import numpy as np
import math
import matplotlib.pyplot as plt

def RK2(f, t_span, c0, n=100):
    dt= (t_span[1]-t_span[0])/n
    t= np.linspace(t_span[0], t_span[1], n+1)
    c=np.zeros(n+1)
    c[0]=c0
    for i in range(n):
        k1=f(t[1], c[1])
        k2=f(t[i+1], c[1]+k1*dt)
        c[i+1]=c[1]+dt*0.5*(k1+k2)
    return t,c
t, c = RK2(lambda t,c: -1*c, [0,2], 1, 10)
print(np.vstack([t,c]).T)

plt.plot(t,c, marker= '*')
plt.show()

#midpoint method
def midpoint_method(f, t_span, c0, n=100):
    dt = (t_span[1] - t_span[0]) / n
    t = np.linspace(t_span[0], t_span[1], n+1)
    c = np.zeros(n+1)
    c[0] = c0

    for i in range(n):
        k1 = f(t[i], c[i])
        k2 = f(t[i] + 0.5*dt, c[i] + 0.5*k1*dt)
        c[i+1] = c[i] + dt*k2

    return t, c
t, c = midpoint_method(lambda t,c: -1*c, [0,2], 1, 10)
print(np.vstack([t,c]).T)

plt.plot(t,c, marker= '*')
plt.show()

#runge-kutta 4
def RK4(f, t_span, c0, n=100):
    dt= (t_span[1]-t_span[0])/n
    t= np.linspace(t_span[0], t_span[1], n+1)
    c=np.zeros(n+1)
    c[0]=c0
    for i in range(n):
        k1=f(t[1], c[1])
        k2=f(t[i+1] +0.5*dt, c[1]+0.5*k1*dt)
        k3=f(t[i+1] +0.5*dt, c[1]+0.5*k2*dt)
        k4=f(t[i+1], c[1]+k3*dt)
        c[i+1]=c[1]+dt*k2
    return t,c

t, c = RK4(lambda t,c: -1*c, [0,2], 1, 10)
print(np.vstack([t,c]).T)

plt.plot(t,c, marker= '*')
plt.show()

#RK45 is the default method in scipy.integrate.solve_ivp
