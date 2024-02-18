import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def system_ode(t, x,alpha,beta,delta,gamma):
    dxdt = np.zeros(2)
    dxdt[0] = alpha*x[0] - beta*x[0]*x[1]
    dxdt[1] = delta*x[0]*x[1] - gamma * x[1]
    return dxdt

x_init = [0.9,1.4]
tspan = [0,10]
sol = scipy.integrate.solve_ivp(system_ode, tspan, x_init, args=(2/3,4/3,1.0,1.0))


fig, ax = plt.subplots(1,2)
ax[0].plot(sol.t, sol.y[0])

ax[1].plot(sol.t, sol.y[1])


plt.show()