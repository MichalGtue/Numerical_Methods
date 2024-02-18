import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,10,10000, endpoint=True)


def func(x):
    if x>2 and x<4:
        return 1
    else:
        return 0
    

y = []

for i in range(len(x)):
    y.append(func(x[i]))

fig = plt.figure(figsize=(10,5))

plt.plot(x,y, 'red', linewidth=2)



plt.xlim(0,10)
plt.ylim(0,1)
plt.grid(which='major', linewidth=3)
plt.minorticks_on()
plt.grid(which='minor', linewidth=0.5)

plt.show()