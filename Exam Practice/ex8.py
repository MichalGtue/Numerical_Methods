import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def func(x,y,a):
    dydx = np.zeros(2)
    A = np.array([[-2, 1], [a-1, -a]])
    g = np.array([2*np.sin(x),a*(np.cos(x)-np.sin(x))])
    dydx = A@y + g
    return dydx



#[-2,1],[1-1,-2]

tspan = [0,10]
x_init = [2,3]
sol = solve_ivp(func, tspan, x_init, args=(2,), rtol=1e-12)

x_plot = np.linspace(0,10,100)
y1 = 2*np.exp(-x_plot) + np.sin(x_plot)
y2 = 2*np.exp(-x_plot) + np.cos(x_plot)

fig = plt.subplots(1,2)

ax1 = plt.subplot(1,2,1)
ax1.plot(sol.t,sol.y[0], label='y1')
ax1.plot(x_plot, y1, label='exact')
plt.legend()
ax2 = plt.subplot(1,2,2)
ax2.plot(sol.t,sol.y[1], label='y2')
ax2.plot(x_plot, y2, label='exact')


plt.legend()
plt.show()