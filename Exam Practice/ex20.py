import numpy as np
import matplotlib.pyplot as plt
import random

def randomxy():
    rand_int = random.random() * 2 * np.pi
    x = np.cos(rand_int) 
    y = np.sin(rand_int)
    return x, y

x = [0]
y = [0]



for i in range(200):
    sol = randomxy()
    x.append(x[-1] + sol[0])
    y.append(y[-1] + sol[1])


plt.plot(x,y)


plt.show()