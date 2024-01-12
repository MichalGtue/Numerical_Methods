import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

def func(x, y, **param):
    c, q = y
    f = np.zeros(y.size)
    f[0] = -q/param['Diff']
    f[1] = -param['kR']*c

    return f


def function(q0, **param):
    sol = solve_ivp(lambda x,y: func(x,y,**param), [0, param['delta']], [param['cAiL'], q0])
    return sol.y[0,-1]
param = {'cAiL': 1.0, 'Diff': 1e-8, 'kR': 10, 'delta': 1e-4, 'N':100}

print(f'ce: {function(0.0003, **param)}')

sol = root_scalar(lambda x: function(x, **param), method='brentq', bracket=[0,1], xtol=1e-15, rtol=1e-15)
q0 = sol.root
print(f'q-:{q0}')

sol = solve_ivp(lambda x,y: func(x,y,**param), [0, param['delta']], [param['cAiL'], q0])
x = sol.t
y = sol.y

fig, ax = plt.subplots()
ax.plot(x,y[0])
plt.show()






