import numpy as np
import matplotlib.pyplot as plt

xlsit=[0]
ylist=[0]

for i in range(60000):
    rand = np.random.rand()
    x = xlsit[-1]
    y = ylist[-1]
    if rand <0.01:
        xlsit.append(0)
        ylist.append(0.16*y)
    elif rand <0.85+0.01:
        xlsit.append(0.85*x + 0.04*y)
        ylist.append(-0.04*x+0.85*y+1.6)
    elif rand < 0.85+0.01+0.07:
        xlsit.append(0.2*x-0.26*y)
        ylist.append(0.23*x+0.22*y+1.6)
    else:
        xlsit.append(-0.15*x + 0.28*y)
        ylist.append(0.26*x+0.24*y+0.44)


plt.plot(xlsit, ylist, '.')

plt.show()