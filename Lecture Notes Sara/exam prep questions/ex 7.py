import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def vdpol(x,y,mu):
    y1,y2=y
    y1prime = y2
    y2prime = mu*(1-y1**2)*y2-y1

    return [y1prime, y2prime]


for mu in [2,5,50]:
    sol = solve_ivp(vdpol, [0,15], [2,0], args=(mu,))
    plt.plot(sol.t,sol.y[0,:],label=f'$y_1$, mu={mu}')
    plt.plot(sol.t,sol.y[1,:],label=f'$y_2$, mu={mu}')


plt.legend()
plt.xlabel('X [-]')
plt.ylabel('Y [-]')
plt.tight_layout()
plt.show()