import numpy as np
import matplotlib.pyplot as plt

def func(t,T):
    return np.exp(-t/T) * np.sin(t)


x_plot = np.linspace(0,10, 100)

y1 = func(x_plot, 1)
y2 = func(x_plot, 2)
y3 = func(x_plot, 4)
y4 = func(x_plot, 8)


plt.plot(x_plot, y1, label='T=1')
plt.plot(x_plot, y2, label='T=2')
plt.plot(x_plot, y3, label='T=4')
plt.plot(x_plot, y4, label='T=8')

plt.legend()
plt.show()

