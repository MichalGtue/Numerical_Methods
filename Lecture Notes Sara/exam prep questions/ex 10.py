import numpy as np
import matplotlib.pyplot as plt
N=1000
t=np.linspace(0,1,1000)
y=np.zeros_like(t)

for i,t_value in enumerate(t):
    y_value = 2/3
    for k in range(1, N+1):
        y+=(-1)**(k+1)+4/(k**2*np.pi**2)*np.cos(k*np.pi*t_value)
    y[i]=y_value

plt.plot(t,y)
plt.show()