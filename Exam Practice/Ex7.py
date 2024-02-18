import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

def fun(t,x):
    dydt = np.zeros(2)
    dydt[0] = x[1]
    dydt[1] = 2*(1-x[0]**2)*x[1] - x[0]
    return dydt
def fun1(t,x):
    dydt = np.zeros(2)
    dydt[0] = x[1]
    dydt[1] = 5*(1-x[0]**2)*x[1] - x[0]
    return dydt

def fun2(t,x):
    dydt = np.zeros(2)
    dydt[0] = x[1]
    dydt[1] = 50*(1-x[0]**2)*x[1] - x[0]
    return dydt

xinit = [2,0]
tspan = [0,15]
sol = scipy.integrate.solve_ivp(fun, tspan, xinit, rtol=1e-12)
sol1 = scipy.integrate.solve_ivp(fun1, tspan, xinit, rtol=1e-12)
sol2 = scipy.integrate.solve_ivp(fun2, tspan, xinit, rtol=1e-12)


fig = plt.subplots()

ax1 = plt.subplot(1,2,1)
ax1.plot(sol.t, sol.y[0],  label='mu=2')
ax1.plot(sol1.t, sol1.y[0],label='mu=5')
ax1.plot(sol2.t, sol2.y[0],label='mu=50')

ax2 = plt.subplot(1,2,2)
ax2.plot(sol.t, sol.y[1],  label='mu=2')
ax2.plot(sol1.t, sol1.y[1],label='mu=5')
ax2.plot(sol2.t, sol2.y[1],label='mu=50')
plt.legend()
plt.show()