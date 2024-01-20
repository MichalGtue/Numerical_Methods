import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(0,3.6,1000, endpoint=False)


def func(x):
    if x<=2 :
        return -3.5*x**2 + 4.5*x + 5
    if x<=3 :
        return -8*x**2+ 40*x - 48
    if x<=3.4 :
        return -15*x**2 + 96*x - 153
    if x<=3.6 :
        return -10*x**2 +70*x -122.4


y = []

for i in range(len(x)):
    y.append(func(x[i]))



plt.plot(x,y)


plt.show()