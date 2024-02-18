import numpy as np
import matplotlib.pyplot as plt


def y(t, it):
 
    total = 0
    for k in range(it):
        total = total + (-1)**(k+2) + 4/((k+1)**2 * np.pi*2) * np.cos((k+1)*np.pi*t)
    return 2/3 + total


x = np.linspace(0,1, endpoint=True)

y_plot = y(x,100)


plt.plot(x,y_plot)


plt.show()