import matplotlib.pyplot as plt
import numpy as np

def func(x,t):
    return np.exp(-(x)/t)*np.sin(x)

x= np.linspace(0,20, 300)

T = [1,2,4,8]
for i in range(len(T)):
    plt.plot(x, func(x,T[i]),label=f'T={T[i]}')
plt.xlabel('t')
plt.ylabel('v(t)')
plt.legend()

plt.show()