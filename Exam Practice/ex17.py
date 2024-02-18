import numpy as np
import matplotlib.pyplot as plt


x_list = np.linspace(0,100,100,dtype=int, endpoint=True)


def func(y,i):
    return y + (2/3)**(i-1)

y=[1]

for i in range(len(x_list)-1):
    y.append(func(y[-1],x_list[i+1]))


plt.plot(x_list,y)


plt.show()