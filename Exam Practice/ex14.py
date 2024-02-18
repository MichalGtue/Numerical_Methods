import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0,10,1000)


y=[]

for i in range(len(x)):
    if x[i]<2:
        y.append(0)
    if x[i]>2 and x[i]<4:
        y.append(1)
    if x[i]>4:
        y.append(0)


fig = plt.figure(figsize=(10,6))
plt.plot(x,y, 'red', linewidth=2)
plt.grid(True,  'major', linestyle='-', linewidth=1)
plt.grid(True,  'minor', linestyle='dotted', linewidth=0.5)

plt.minorticks_on()
plt.xlabel('Time [s]')
plt.ylabel('Signal [-]')
plt.xlim(0,10)
plt.ylim(0,1)
plt.show()