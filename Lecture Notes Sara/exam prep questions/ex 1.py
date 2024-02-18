import scipy.integrate as sp
import numpy as np
import matplotlib.pyplot as plt


def system_ode(t, Xin, params):
    alpha, beta, gamma, delta = params.values()

    x,y= Xin

    dxdt = alpha*x - beta*x*y
    dydt = delta*x*y - gamma*y

    return [dxdt, dydt]

par= {'alpha':2/3, 'beta':4/3, 'gamma':1.0, 'delta':1.0}

tspan = [0,50]
t_eval = np.linspace(tspan[0], tspan[1], 1000)
x0= 0.9
y0= 1.4
init=[x0,y0]

sol = sp.solve_ivp(system_ode, tspan, init, t_eval=t_eval, args=(par,))

t= sol.t
x=sol.y
plt.plot(t,x[0,:],t,x[1,:])
plt.xlabel('time [s]')
plt.ylabel('x,y [a.u.]')
plt.tight_layout()
plt.show()

